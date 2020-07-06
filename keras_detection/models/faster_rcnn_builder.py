import json
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import tensorflow as tf

import keras_detection.models.box_detector as box_detector
import keras_detection.ops.tflite_ops as tflite_ops
import keras_detection.tasks as dt
from keras_detection.api import (
    LabelDescription,
    ModelMetadata,
    OutputTensorType,
    TaskType,
)
from keras_detection.backbones.base import Backbone
from keras_detection.datasets.datasets_ops import prepare_dataset
import keras_detection.ops.tflite_metadata as tflite_metadata_ops
from keras_detection.layers.roi_align import ROIAlignLayer
from keras_detection.structures import ImageData
from keras_detection.targets.base import FeatureMapDesc
from keras_detection.targets import feature_map_sampling as fm_sampling

LOGGER = tf.get_logger()
keras = tf.keras
Lambda = keras.layers.Lambda
K = keras.backend


class FasterRCNNBuilder:
    def __init__(
        self,
        backbone: Backbone,
        rpn_objectness_task: dt.PredictionTaskDef,
        rpn_box_shape_task: dt.PredictionTaskDef,
        rcnn_tasks: List[dt.PredictionTaskDef],
    ):
        self.rcnn_tasks = rcnn_tasks
        self.backbone = backbone
        self.built = False

        print(backbone.output_shapes)
        # RPN
        self.rpn = RPN(
            image_input_shape=self.input_shape,
            feature_maps_shapes=backbone.output_shapes,
            rpn_objectness_task=rpn_objectness_task,
            rpn_box_shape_task=rpn_box_shape_task,
            # name="RPN",
        )
        self.roi_sampling = ROISamplingLayer(num_samples=64, crop_size=(7, 7))

        self.rcnn = RCNN(
            image_input_shape=self.input_shape,
            feature_maps_shapes=backbone.output_shapes,
            tasks=rcnn_tasks,
            # name="RCNN",
        )

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self.backbone.input_shape

    @property
    def input_name(self) -> str:
        return "image"

    def build(self, batch_size: int, is_training: bool = True):

        input_image, feature_maps = self.build_backbone(
            batch_size=batch_size, is_training=is_training, quantized=False
        )
        assert (
            len(feature_maps) == 1
        ), "Case when there are more FMs than 1 is not supported yet"

        rpn_outputs = self.rpn(feature_maps)
        rpn_predictions_raw = self.rpn.predictions_to_dict(
            rpn_outputs, postprocess=False
        )
        sampled_anchors = None
        rpn_scores = None
        if is_training:
            rpn_targets_inputs = self.rpn.get_targets_input_tensors(
                batch_size=batch_size
            )
            rpn_boxes, rpn_loss_map = self.rpn.get_rpn_loss_map(
                rpn_outputs, rpn_targets_inputs
            )
            crops, crops_boxes, crops_indices = self.roi_sampling(
                [feature_maps[0], rpn_boxes, rpn_loss_map], training=is_training
            )
            print("crops:", crops.shape)
            rcnn_targets_inputs = self.rcnn.get_targets_input_tensors(
                batch_size=batch_size
            )
            rcnn_targets = self.roi_sampling.sample_targets_tensors(
                rcnn_targets_inputs, indices=crops_indices
            )
        else:
            rpn_targets_inputs = {}
            rcnn_targets_inputs = {}
            rpn_scores, rpn_boxes, sampled_anchors, sampled_indices = self.rpn.sample_proposal_boxes(
                rpn_outputs, self.roi_sampling.num_samples
            )
            print(sampled_indices)
            crops = self.roi_sampling.roi_align([feature_maps[0], rpn_boxes])

        rcnn_outputs = self.rcnn([crops])
        rcnn_predictions_raw = self.rcnn.predictions_to_dict(
            rcnn_outputs, postprocess=False
        )

        if is_training:
            rcnn_predictions_raw["rcnn/crops"] = crops
            rcnn_predictions_raw["rcnn/crops_boxes"] = crops_boxes
            rcnn_predictions_raw["rcnn/crops_indices"] = crops_indices

        if sampled_anchors is not None:
            rcnn_predictions_raw["rcnn/anchors"] = sampled_anchors
            rcnn_predictions_raw["rcnn/scores"] = rpn_scores
            rcnn_predictions_raw["rcnn/boxes"] = rpn_boxes

        outputs = {}
        # outputs["feature_maps/fm0"] = feature_maps
        outputs.update(rpn_predictions_raw)
        outputs.update(rcnn_predictions_raw)

        outputs = {
            name: keras.layers.Lambda(lambda x: x, name=name)(tensor)
            for name, tensor in outputs.items()
        }

        rcnn_model = keras.Model(
            inputs=[
                input_image,
                *rpn_targets_inputs.values(),
                *rcnn_targets_inputs.values(),
            ],
            outputs=outputs,
        )

        if is_training:
            self.add_rcnn_loss(rcnn_model, rcnn_targets, rcnn_predictions_raw)

        return rcnn_model

    def add_rcnn_loss(
        self,
        model: keras.Model,
        rcnn_targets: Dict[str, tf.Tensor],
        rcnn_predictions: Dict[str, tf.Tensor],
    ):

        weights = self.rcnn.get_losses_weights()

        for key, loss_class in self.rcnn.get_losses().items():
            print("Adding loss: ", key)
            y_true = rcnn_targets[key]
            y_pred = rcnn_predictions[key]
            loss = loss_class.call(y_true=y_true, y_pred=y_pred)
            loss = loss * weights[key]
            model.add_loss(tf.reduce_mean(loss))
            model.add_metric(tf.reduce_mean(loss), name=key, aggregation="mean")

    def build_backbone(
        self, batch_size: Optional[int], is_training: bool, quantized: bool
    ):

        input_image = keras.Input(
            shape=self.input_shape, name="image", batch_size=batch_size
        )

        LOGGER.info(f"Input image: {input_image.shape}")
        LOGGER.info(f"Processing backbone: {self.backbone}")

        inputs = self.backbone.preprocess_images(input_image, is_training)
        feature_maps = self.backbone.forward(
            inputs, is_training=is_training, quantized=quantized
        )
        if isinstance(feature_maps, tf.Tensor):
            # keras removes list when there is only ony feature map
            feature_maps = [feature_maps]

        return input_image, feature_maps

    def prepare_dataset(self, dataset):
        def prepare_fn(batch_data):
            features, targets = self.get_build_training_targets_fn(
                self.rpn.rpn_fm_prediction_tasks + self.rcnn.fm_prediction_tasks
            )(batch_data)
            for name in list(targets.keys()):
                features[f"inputs/{name}"] = targets[name]
                if "rcnn" in name:
                    del targets[name]

            return features, targets

        return dataset.map(prepare_fn)

    def get_build_training_targets_fn(
        self, fm_prediction_tasks: List[dt.FeatureMapPredictionTasks]
    ):
        def prepare_dataset_fn(batch_data: Dict[str, Any]):
            batch_data: ImageData[tf.Tensor] = ImageData.from_dict(batch_data)
            batch_frame = batch_data.labels
            labels = {}

            for fmt in fm_prediction_tasks:
                for name, target in fmt.get_targets(batch_frame).items():
                    labels[name] = target
            return batch_data.features.to_dict(), labels

        return prepare_dataset_fn


class RPN(keras.layers.Layer):
    def __init__(
        self,
        image_input_shape: Tuple[int, int, int],
        feature_maps_shapes: List[Tuple[int, int, int]],
        rpn_objectness_task: dt.PredictionTaskDef,
        rpn_box_shape_task: dt.PredictionTaskDef,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_input_shape = image_input_shape
        self.rpn_box_shape_task = rpn_box_shape_task
        self.rpn_objectness_task = rpn_objectness_task
        self.rpn_fm_descs: Optional[List[FeatureMapDesc]] = None
        self.rpn_fm_prediction_tasks: Optional[
            List[dt.FeatureMapPredictionTasks]
        ] = None

        self.build_rpn_heads(feature_maps_shapes)

    @property
    def rpn_tasks_defs(self) -> List[dt.PredictionTaskDef]:
        return [self.rpn_objectness_task, self.rpn_box_shape_task]

    def get_metrics(self) -> Dict[str, dt.MetricType]:
        metrics = {}
        for pt in self.rpn_fm_prediction_tasks:
            metrics.update(pt.get_metrics())
        return metrics

    def get_losses(self) -> Dict[str, dt.LossType]:

        losses = {}
        for pt in self.rpn_fm_prediction_tasks:
            losses.update(pt.get_losses())
        return losses

    def get_losses_weights(self) -> Dict[str, float]:

        losses_weights = {}
        for pt in self.rpn_fm_prediction_tasks:
            losses_weights.update(pt.get_losses_weights())
        return losses_weights

    def get_model_compile_args(self) -> Dict[str, Dict[str, Any]]:
        args = {
            "loss": self.get_losses(),
            "loss_weights": self.get_losses_weights(),
            "metrics": self.get_metrics(),
        }
        return args

    def call(self, feature_maps: List[tf.Tensor], training: bool = False, **kwargs):

        task_names = [t.name for t in self.rpn_tasks_defs]
        LOGGER.info(f"Processing RPN feature maps for tasks: {task_names}")
        fm_outputs = []
        for feature_map, fm_tasks in zip(feature_maps, self.rpn_fm_prediction_tasks):
            LOGGER.info(f" Processing RPN feature map ({fm_tasks.name})")
            fm_outs = fm_tasks.get_outputs(
                feature_map, is_training=training, quantized=False
            )
            fm_outputs += fm_outs

        return fm_outputs

    def build_rpn_heads(self, feature_maps_shapes: List[Tuple[int, int, int]]):

        self.rpn_fm_descs = [
            FeatureMapDesc(
                fm_height=fm[0],
                fm_width=fm[1],
                image_height=self.image_input_shape[0],
                image_width=self.image_input_shape[1],
            )
            for fm in feature_maps_shapes
        ]

        self.rpn_fm_prediction_tasks = []
        for fm_id, feature_map in enumerate(feature_maps_shapes):
            # each feature map has its own head, they are not shared
            fm_name = self.rpn_fm_descs[fm_id].fm_name
            fm_name = f"rpn/{fm_name}"

            tasks = []
            for task in self.rpn_tasks_defs:
                fm_task = dt.PredictionTask.from_task_def(f"head/{task.name}", task)
                tasks.append(fm_task)

            fm_tasks = dt.FeatureMapPredictionTasks(
                fm_name, self.rpn_fm_descs[fm_id], tasks
            )
            self.rpn_fm_prediction_tasks.append(fm_tasks)

    def get_rpn_loss_map(
        self, rpn_predictions: List[tf.Tensor], targets: Dict[str, tf.Tensor]
    ):

        rpn_predictions_raw = self.predictions_to_dict(
            rpn_predictions, postprocess=False
        )
        loss_weights = self.get_losses_weights()
        losses = []

        for key, loss_class in self.get_losses().items():
            loss_class.per_anchor_loss = True
            loss = loss_class.call(y_true=targets[key], y_pred=rpn_predictions_raw[key])
            weight = loss_weights[key]
            losses.append(loss * weight)
            loss_class.per_anchor_loss = False

        rpn_loss_map = tf.add_n(losses)

        objectness, box_shape = rpn_predictions
        fm = self.rpn_fm_prediction_tasks[0]
        boxes = self.rpn_box_shape_task.target_builder.postprocess_predictions(
            fm.fm_desc, box_shape
        )
        boxes = self.rpn_box_shape_task.target_builder.to_tf_boxes(boxes)
        return boxes, rpn_loss_map

    def sample_proposal_boxes(self, rpn_predictions: List[tf.Tensor], num_samples: int):
        objectness, box_shape = rpn_predictions
        fm = self.rpn_fm_prediction_tasks[0]
        boxes = self.rpn_box_shape_task.target_builder.postprocess_predictions(
            fm.fm_desc, box_shape
        )

        boxes = self.rpn_box_shape_task.target_builder.to_tf_boxes(boxes)

        anchors = self.rpn_box_shape_task.target_builder.postprocess_predictions(
            fm.fm_desc, tf.zeros_like(box_shape)
        )

        batch_size = boxes.shape[0]
        sampled_boxes = []
        sampled_anchors = []
        sampled_scores = []
        sampled_indices = []
        for i in range(batch_size):
            batch_boxes = tf.reshape(boxes[i, ...], [-1, 4])
            batch_scores = tf.reshape(objectness[i, ...], [-1])
            batch_indices = tf.image.non_max_suppression(
                batch_boxes,
                batch_scores,
                fm.fm_desc.fm_width * fm.fm_desc.fm_height,
                iou_threshold=0.35,
                score_threshold=0.5,
            )
            selected_boxes = tf.gather(batch_boxes, batch_indices)
            selected_boxes = pad_or_slice(selected_boxes, num_samples)

            selected_scores = tf.gather(batch_scores, batch_indices)
            selected_scores = tf.reshape(selected_scores, [-1, 1])
            selected_scores = pad_or_slice(selected_scores, num_samples)

            batch_anchors = tf.gather(
                tf.reshape(anchors[i, ..., 2:], [-1, 2]), batch_indices
            )
            batch_anchors = pad_or_slice(batch_anchors, num_samples)

            sampled_indices.append(batch_indices)
            sampled_anchors.append(batch_anchors)
            sampled_boxes.append(selected_boxes)
            sampled_scores.append(selected_scores)

        sampled_boxes = tf.stack(sampled_boxes, axis=0)
        sampled_anchors = tf.stack(sampled_anchors, axis=0)
        sampled_scores = tf.stack(sampled_scores, axis=0)
        sampled_indices = tf.stack(sampled_indices, axis=0)
        # print("sampled_boxes (test):", sampled_boxes)
        # print("sampled_anchors (test):", sampled_anchors)
        # print("sampled_anchors (test):", sampled_scores)
        return sampled_scores, sampled_boxes, sampled_anchors, sampled_indices

    def get_targets_input_tensors(
        self, batch_size: Optional[int] = None, prefix: str = "inputs/"
    ) -> Dict[str, tf.Tensor]:
        names = self.get_outputs_names()
        shapes = self.get_targets_outputs_shapes()
        inputs = {}
        for name, shape in zip(names, shapes):
            inputs[name] = keras.Input(
                shape=shape[1:], name=f"{prefix}{name}", batch_size=batch_size
            )
        return inputs

    def get_targets_outputs_shapes(self) -> List[Tuple[Optional[int], int, int, int]]:
        outputs_names = []
        for pt in self.rpn_fm_prediction_tasks:
            outputs_names += pt.targets_outputs_shapes
        return outputs_names

    def get_outputs_names(self) -> List[str]:
        outputs_names = []
        for pt in self.rpn_fm_prediction_tasks:
            outputs_names += pt.outputs_names
        return outputs_names

    def predictions_to_dict(
        self, predictions: List[tf.Tensor], postprocess: bool = False
    ) -> Dict[str, tf.Tensor]:

        if isinstance(predictions, tf.Tensor) or isinstance(predictions, np.ndarray):
            # keras removes list when there is only ony feature map
            predictions = [predictions]

        predictions_dict = {
            name: tensor for name, tensor in zip(self.get_outputs_names(), predictions)
        }
        if not postprocess:
            return predictions_dict
        else:
            postprocessed_dict = {}
            for fm in self.rpn_fm_prediction_tasks:
                for task in fm.tasks:
                    name = fm.fm_task_name(task)
                    outputs = task.postprocess(fm.fm_desc, predictions_dict[name])
                    postprocessed_dict[name] = outputs
            return postprocessed_dict


class ROISamplingLayer(keras.layers.Layer):
    def __init__(self, num_samples: int, crop_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.roi_align = ROIAlignLayer(crop_size=crop_size)

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], training: bool = True
    ):
        feature_map, boxes, rpn_loss_map = inputs
        indices = fm_sampling.scores_to_gather_indices(rpn_loss_map, self.num_samples)
        # fm_sampling.sample_feature_map(tf.expand_dims(rpn_loss_map, -1), indices)
        sampled_boxes = fm_sampling.sample_feature_map(boxes, indices)
        indices = tf.stop_gradient(indices)
        sampled_boxes = tf.stop_gradient(sampled_boxes)

        # # naive sampling
        # crops = fm_sampling.sample_feature_map(feature_map, indices)
        # num_channels = feature_map.shape[-1]
        # crops = tf.reshape(crops, [-1, 1, 1, num_channels])

        crops = self.roi_align([feature_map, sampled_boxes])

        return crops, sampled_boxes, indices

    def sample_targets_tensors(self, targets: Dict[str, tf.Tensor], indices: tf.Tensor):
        crops_targets = {}
        for key, target in targets.items():
            target = fm_sampling.sample_feature_map(target, indices)
            nc = target.shape[-1]
            batch_size = target.shape[0]
            crops_targets[key] = tf.reshape(
                target, [batch_size * self.num_samples, 1, 1, nc]
            )
        return crops_targets


class RCNN(keras.layers.Layer):
    def __init__(
        self,
        image_input_shape: Tuple[int, int, int],
        feature_maps_shapes: List[Tuple[int, int, int]],
        tasks: List[dt.PredictionTaskDef],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_input_shape = image_input_shape
        self.tasks = tasks
        self.fm_descs: Optional[List[FeatureMapDesc]] = None
        self.fm_prediction_tasks: Optional[List[dt.FeatureMapPredictionTasks]] = None

        self.build_heads(feature_maps_shapes)

    def get_metrics(self) -> Dict[str, dt.MetricType]:
        metrics = {}
        for pt in self.fm_prediction_tasks:
            metrics.update(pt.get_metrics())
        return metrics

    def get_losses(self) -> Dict[str, dt.LossType]:

        losses = {}
        for pt in self.fm_prediction_tasks:
            losses.update(pt.get_losses())
        return losses

    def get_losses_weights(self) -> Dict[str, float]:

        losses_weights = {}
        for pt in self.fm_prediction_tasks:
            losses_weights.update(pt.get_losses_weights())
        return losses_weights

    def get_model_compile_args(self) -> Dict[str, Dict[str, Any]]:
        args = {
            "loss": self.get_losses(),
            "loss_weights": self.get_losses_weights(),
            "metrics": self.get_metrics(),
        }
        return args

    def call(self, feature_maps: List[tf.Tensor], training: bool = False, **kwargs):

        task_names = [t.name for t in self.tasks]
        LOGGER.info(f"Processing RCNN feature maps for tasks: {task_names}")
        fm_outputs = []
        for feature_map, fm_tasks in zip(feature_maps, self.fm_prediction_tasks):
            LOGGER.info(f" Processing RCNN feature map ({fm_tasks.name})")
            fm_outs = fm_tasks.get_outputs(
                feature_map, is_training=training, quantized=False
            )
            fm_outputs += fm_outs

        return fm_outputs

    def build_heads(self, feature_maps_shapes: List[Tuple[int, int, int]]):

        self.fm_descs = [
            FeatureMapDesc(
                fm_height=fm[0],
                fm_width=fm[1],
                image_height=self.image_input_shape[0],
                image_width=self.image_input_shape[1],
            )
            for fm in feature_maps_shapes
        ]

        self.fm_prediction_tasks = []
        for fm_id, feature_map in enumerate(feature_maps_shapes):
            # each feature map has its own head, they are not shared
            fm_name = self.fm_descs[fm_id].fm_name
            fm_name = f"rcnn/{fm_name}"

            tasks = []
            for task in self.tasks:
                fm_task = dt.PredictionTask.from_task_def(
                    f"rcnn/head/{task.name}", task
                )
                tasks.append(fm_task)

            fm_tasks = dt.FeatureMapPredictionTasks(
                fm_name, self.fm_descs[fm_id], tasks
            )
            self.fm_prediction_tasks.append(fm_tasks)

    def get_targets_input_tensors(
        self, batch_size: Optional[int] = None, prefix: str = "inputs/"
    ) -> Dict[str, tf.Tensor]:
        names = self.get_outputs_names()
        shapes = self.get_targets_outputs_shapes()
        inputs = {}
        for name, shape in zip(names, shapes):
            inputs[name] = keras.Input(
                shape=shape[1:], name=f"{prefix}{name}", batch_size=batch_size
            )
        return inputs

    def get_targets_outputs_shapes(self) -> List[Tuple[Optional[int], int, int, int]]:
        outputs_names = []
        for pt in self.fm_prediction_tasks:
            outputs_names += pt.targets_outputs_shapes
        return outputs_names

    def get_outputs_names(self) -> List[str]:
        outputs_names = []
        for pt in self.fm_prediction_tasks:
            outputs_names += pt.outputs_names
        return outputs_names

    def predictions_to_dict(
        self, predictions: List[tf.Tensor], postprocess: bool = False
    ) -> Dict[str, tf.Tensor]:

        if isinstance(predictions, tf.Tensor) or isinstance(predictions, np.ndarray):
            # keras removes list when there is only ony feature map
            predictions = [predictions]

        predictions_dict = {
            name: tensor for name, tensor in zip(self.get_outputs_names(), predictions)
        }
        if not postprocess:
            return predictions_dict
        else:
            postprocessed_dict = {}
            for fm in self.fm_prediction_tasks:
                for task in fm.tasks:
                    name = fm.fm_task_name(task)
                    outputs = task.postprocess(fm.fm_desc, predictions_dict[name])
                    postprocessed_dict[name] = outputs
            return postprocessed_dict


# @tf.function
def pad_or_slice(
    tensor: tf.Tensor, target_rows: int, constant_values: float = 0
) -> tf.Tensor:
    """
    Pad first axis with zeros or take first target_rows
    Args:
        tensor: tensor of shape [num_rows, num_features]
        target_rows:
    Returns:
        tensor [target_rows, num_features]
    """
    feats_shapes = tensor.shape.as_list()[1:]

    tensor = tf.pad(
        tensor, [[0, target_rows], [0, 0]], "CONSTANT", constant_values=constant_values
    )
    tensor = tensor[:target_rows, :]
    return tf.reshape(tensor, [target_rows, *feats_shapes])
