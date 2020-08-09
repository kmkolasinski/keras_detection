from typing import Any, Dict, Optional
from typing import List

import numpy as np
import tensorflow as tf

import keras_detection.losses as losses
from keras_detection import FeatureMapPredictionTarget
from keras_detection.losses import FeatureMapPredictionTargetLoss, MulticlassTarget
from keras_detection.models.box_detector import BoxDetector, BoxDetectionOutput
from keras_detection.modular.backbones.core import FeatureMapDescEstimator
from keras_detection.modular.backbones.fpn import FPN
from keras_detection.modular.backbones.resnet import ResNet
from keras_detection.modular.core import (
    NodeLoss,
    NeuralGraph,
    Node,
    ImageInputNode,
    LabelsInputNode,
    KerasGraph,
)
from keras_detection.modular.core import TrainableModule
from keras_detection.modular.heads.heads import SingleConvHead, Head
from keras_detection.structures import FeatureMapDesc, LabelsFrame
from keras_detection.targets.box_objectness import BoxCenterIgnoreMarginObjectnessTarget
from keras_detection.targets.box_shape import BoxShapeTarget

keras = tf.keras


class FeatureMapNodeLoss(NodeLoss):
    def __init__(
        self,
        inputs: List[str],
        name: str,
        target_assigner: FeatureMapPredictionTarget,
        loss: FeatureMapPredictionTargetLoss,
        weight: float = 1.0,
    ):
        super().__init__(inputs, weight=weight)
        self.box_shape_ta = target_assigner
        self.loss = loss
        self.loss_tracker = keras.metrics.Mean(name=name)
        self.fm_desc: FeatureMapDesc = None

    def call(
        self,
        fm_desc: FeatureMapDesc,
        batch_frame: LabelsFrame,
        predicted: tf.Tensor,
        per_anchor_loss: bool = False,
    ) -> tf.Tensor:
        targets = self.box_shape_ta.get_targets_tensors(fm_desc, batch_frame)
        self.fm_desc = fm_desc
        return self.loss.call(targets, predicted, per_anchor_loss=per_anchor_loss)

    def decode_predictions(self, outputs):
        return self.box_shape_ta.postprocess_predictions(self.fm_desc, outputs)


class BoxShapeLoss(FeatureMapNodeLoss):
    def __init__(self, inputs: List[str], weight: float = 5.0):
        box_shape_ta = BoxShapeTarget(use_tf_format=True)
        super().__init__(
            inputs,
            "box_shape",
            box_shape_ta,
            losses.L1Loss(box_shape_ta),
            weight=weight,
        )


class BoxObjectnessLoss(FeatureMapNodeLoss):
    def __init__(self, inputs: List[str], weight: float = 1.0):
        box_objectness_ta = BoxCenterIgnoreMarginObjectnessTarget(pos_weights=5.0)
        loss = losses.BCELoss(
            box_objectness_ta,
            label_smoothing=0.02,
            smooth_only_positives=True,
            from_logits=False,
        )
        super().__init__(inputs, "objectness", box_objectness_ta, loss, weight=weight)


class ClassesLoss(FeatureMapNodeLoss):
    def __init__(self, inputs: List[str], num_classes: int, weight: float = 1.0):
        classes_ta = MulticlassTarget(num_classes)
        loss = losses.BCELoss(
            classes_ta,
            label_smoothing=0.02,
            smooth_only_positives=False,
            from_logits=False,
        )
        super().__init__(inputs, "classes", classes_ta, loss, weight=weight)


class FeatureMapHead(TrainableModule):
    def __init__(
        self, name: str, head: Head, loss: FeatureMapNodeLoss, *args, **kwargs
    ):
        super().__init__(name=name, *args, **kwargs)
        self.head = head
        self.head_loss = loss

    def call(self, inputs, training: bool = False) -> Dict[str, Any]:
        # Keras does not work well with layers with more than one input
        # especially when we want to save it to saved_model format
        return self._call(*inputs, training=training)

    def _call(
        self,
        feature_map: tf.Tensor,
        feature_map_desc: FeatureMapDescEstimator,
        training: bool = False,
    ) -> Dict[str, Any]:

        predictions = self.head(feature_map, training=training)
        self.head_loss.fm_desc = feature_map_desc
        postprocessed_outputs = self.head_loss.decode_predictions(predictions)
        outputs = {"raw": predictions, f"postprocessed": postprocessed_outputs}
        return outputs

    def as_node(self, inputs: List[str]) -> Node:
        return Node(
            self.name,
            inputs=inputs,
            module=self,
            loss=self.head_loss,
            inputs_as_list=True,
        )


class RetinaDetector(NeuralGraph):
    def __init__(self, num_classes: int, training: bool = True):
        super().__init__()

        image_dim = 224
        self.image_dim = image_dim
        self.num_classes = num_classes

        backbone = ResNet(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1, 1),
            num_last_blocks=2,
        )

        fpn = FPN(
            input_shapes=backbone.get_output_shapes(), depth=64, num_first_blocks=1
        )

        self.add(ImageInputNode())
        self.add(LabelsInputNode())

        self.add(Node("backbone", inputs=["image"], module=backbone))
        self.add(
            Node(
                "backbone/fm0/desc",
                inputs=["image", "backbone/fm0"],
                module=FeatureMapDescEstimator(),
            )
        )
        self.add(
            Node(
                "fpn",
                inputs=backbone.get_output_names("backbone"),
                inputs_as_list=True,
                module=fpn,
            )
        )

        self.add(
            FeatureMapHead(
                "boxes",
                head=SingleConvHead("boxes", 4, activation=None),
                loss=BoxShapeLoss(inputs=["backbone/fm0/desc", "labels", "boxes/raw"]),
            ).as_node(inputs=["fpn/fm0", "backbone/fm0/desc"])
        )

        self.add(
            FeatureMapHead(
                "objectness",
                head=SingleConvHead("objectness", 1, activation="sigmoid"),
                loss=BoxObjectnessLoss(
                    inputs=["backbone/fm0/desc", "labels", "objectness/raw"]
                ),
            ).as_node(inputs=["fpn/fm0", "backbone/fm0/desc"])
        )

        self.add(
            FeatureMapHead(
                "classes",
                head=SingleConvHead("classes", num_classes + 1, activation="sigmoid"),
                loss=ClassesLoss(
                    num_classes=num_classes,
                    inputs=["backbone/fm0/desc", "labels", "classes/raw"],
                ),
            ).as_node(inputs=["fpn/fm0", "backbone/fm0/desc"])
        )

    def as_predictor(self, batch_size: int, weights: str) -> keras.Model:
        image = tf.keras.Input(
            shape=(self.image_dim, self.image_dim, 3),
            batch_size=batch_size,
            name="image",
        )
        model = KerasGraph(
            graph=RetinaDetector(self.num_classes, training=False), name="RetinaNet"
        )
        predictor = keras.Model(image, model({"features": {"image": image}}))
        model.load_weights(weights)
        return predictor


class FPNBoxDetector(BoxDetector):
    def __init__(
        self,
        model: keras.Model,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.35,
    ):
        super().__init__(score_threshold, iou_threshold)
        self.model = model

    def _predict(self, images: np.ndarray, **kwargs) -> List[BoxDetectionOutput]:

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)

        predictions = self.model.predict(images)
        outputs = []
        for i in range(images.shape[0]):
            boxes = predictions["boxes/postprocessed"][i].reshape([-1, 4])
            scores = predictions["objectness/postprocessed"][i].reshape([-1])

            classes_scores = predictions["classes/postprocessed"][i]
            num_classes = classes_scores.shape[-1]
            classes_scores = classes_scores.reshape([-1, num_classes])

            output = BoxDetectionOutput.from_tf_boxes(
                boxes=boxes,
                scores=scores,
                labels=classes_scores.argmax(-1),
                classes_scores=classes_scores,
            )
            outputs.append(output)

        return outputs
