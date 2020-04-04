from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Union

import numpy as np
import tensorflow as tf

import keras_detection.models.box_detector as box_detector
import keras_detection.ops.tflite_ops as tflite_ops
import keras_detection.tasks as dt
from keras_detection.backbones.base import Backbone
from keras_detection.datasets.datasets_ops import prepare_dataset
from keras_detection.structures import ImageData
from keras_detection.targets.base import FeatureMapDesc

LOGGER = tf.get_logger()
keras = tf.keras
Lambda = keras.layers.Lambda


class FPNBuilder:
    def __init__(
        self, backbone: Backbone, tasks: List[dt.PredictionTaskDef], *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.backbone = backbone
        self.tasks = tasks
        self.fm_descs: Optional[List[FeatureMapDesc]] = None
        self.fm_prediction_tasks: Optional[List[dt.FeatureMapPredictionTasks]] = None
        self.built = False
        self.quantized = False

    @property
    def input_shape(self) -> Tuple[int, int, int]:
        return self.backbone.input_shape

    @property
    def input_name(self) -> str:
        return "image"

    @property
    def num_tasks(self) -> int:
        return len(self.tasks)

    @property
    def num_fm_maps(self) -> int:
        self.check_built()
        return len(self.fm_descs)

    def get_outputs_names(self) -> List[str]:
        self.check_built()
        outputs_names = []
        for pt in self.fm_prediction_tasks:
            outputs_names += pt.outputs_names
        return outputs_names

    def get_metrics(self) -> Dict[str, dt.MetricType]:
        self.check_built()
        metrics = {}
        for pt in self.fm_prediction_tasks:
            metrics.update(pt.get_metrics())
        return metrics

    def get_losses(self) -> Dict[str, dt.LossType]:
        self.check_built()
        losses = {}
        for pt in self.fm_prediction_tasks:
            losses.update(pt.get_losses())
        return losses

    def get_losses_weights(self) -> Dict[str, float]:
        self.check_built()
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

    def check_built(self):
        assert self.built, "Model not built. Call build() function first!"

    def build_backbone(self, batch_size: int, is_training: bool, quantized: bool):

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

    def build_heads(self, feature_maps: List[tf.Tensor]) -> None:

        self.fm_descs = [
            FeatureMapDesc(
                fm_height=fm.shape[1],
                fm_width=fm.shape[2],
                image_height=self.input_shape[0],
                image_width=self.input_shape[1],
            )
            for fm in feature_maps
        ]

        self.fm_prediction_tasks = []
        for fm_id, feature_map in enumerate(feature_maps):
            # each feature map has its own head, they are not shared
            fm_name = self.fm_descs[fm_id].fm_name
            tasks = []
            for task in self.tasks:
                fm_task = dt.PredictionTask.from_task_def(
                    f"{fm_name}/{task.name}", task
                )
                tasks.append(fm_task)

            fm_tasks = dt.FeatureMapPredictionTasks(
                fm_name, self.fm_descs[fm_id], tasks
            )
            self.fm_prediction_tasks.append(fm_tasks)

    def build(
        self,
        batch_size: Optional[int] = None,
        name: str = None,
        is_training: bool = False,
        quantized: bool = False,
        initialize_heads: bool = True,
    ) -> keras.Model:

        input_image, feature_maps = self.build_backbone(
            batch_size=batch_size, is_training=is_training, quantized=quantized
        )

        if initialize_heads:
            self.build_heads(feature_maps)
        else:
            self.check_built()

        task_names = [t.name for t in self.tasks]
        LOGGER.info(f"Processing feature maps for tasks: {task_names}")
        fm_outputs = []
        for feature_map, fm_tasks in zip(feature_maps, self.fm_prediction_tasks):
            LOGGER.info(f" Processing feature map ({fm_tasks.name})")
            fm_outs = fm_tasks.get_outputs(
                feature_map, is_training=is_training, quantized=quantized
            )
            fm_outputs.append(fm_outs)

        self.built = True
        outputs = list(chain.from_iterable(fm_outputs))
        return keras.Model(inputs=[input_image], outputs=outputs, name=name)

    def build_quantized(
        self,
        batch_size: Optional[int] = None,
        non_quantized_model_weights: Optional[Union[str, Path]] = None,
        name: str = None,
        is_training: bool = True,
    ) -> keras.Model:

        LOGGER.info(f"Building quantized model with batch_size = {batch_size}")
        if non_quantized_model_weights is not None:
            model = self.build(
                batch_size=batch_size, is_training=is_training, quantized=False
            )
            LOGGER.info(f"Loading weights of base model: {non_quantized_model_weights}")
            model.load_weights(str(non_quantized_model_weights))

        LOGGER.info(f"Building graph with quantization enabled")
        model = self.build(
            batch_size=batch_size,
            is_training=is_training,
            name=name,
            quantized=True,
            initialize_heads=non_quantized_model_weights is None,
        )
        return model

    def postprocess_model(
        self,
        base_model: keras.Model,
        batch_size: Optional[int],
        postprocess_outputs: bool = False,
        merge_feature_maps: bool = False,
        name: str = None,
    ) -> keras.Model:
        self.check_built()

        input_image = keras.Input(
            shape=self.input_shape, name="image", batch_size=batch_size
        )
        predictions = base_model(input_image)
        predictions_dict = self.predictions_to_dict(predictions, postprocess_outputs)
        if merge_feature_maps:
            predictions_dict = self.merge_output_feature_maps(predictions_dict)

        LOGGER.info(f"Export outputs")
        export_outputs = []
        for task_name, outputs in predictions_dict.items():
            if isinstance(outputs, dict):
                for output_name, output in outputs.items():
                    out = Lambda(
                        lambda x: tf.identity(x, name="output"),
                        name=f"{task_name}/{output_name}",
                    )(output)
                    export_outputs.append(out)
                    LOGGER.info(f"- {task_name:20} - {out}")
            elif isinstance(outputs, tf.Tensor):
                out = Lambda(lambda x: tf.identity(x, name="output"), name=task_name)(
                    outputs
                )
                export_outputs.append(out)
                LOGGER.info(f"- {task_name:20} - {out}")
            else:
                raise ValueError(
                    f"Unsupported task output type: {type(outputs)}: {outputs}"
                )

        return keras.Model(inputs=[input_image], outputs=export_outputs, name=name)

    def merge_output_feature_maps(
        self, prediction_dict: Dict[str, tf.Tensor]
    ) -> Dict[str, tf.Tensor]:

        self.check_built()
        LOGGER.info("Merging feature maps")
        per_task_outputs = defaultdict(list)
        for fm_task in self.fm_prediction_tasks:
            for task in fm_task.tasks:
                name = fm_task.fm_task_name(task)
                fm_out = prediction_dict[name]
                if isinstance(fm_out, tf.Tensor):
                    fm_out = reshape_feature_map(fm_out)
                    per_task_outputs[task.name].append(fm_out)
                elif isinstance(fm_out, dict):
                    for name, fm_tensor in fm_out.items():
                        fm = reshape_feature_map(fm_tensor)
                        per_task_outputs[f"{task.name}/{name}"].append(fm)
                else:
                    raise ValueError(
                        f"Unsupported task output type: {type(fm_out)}: {fm_out}"
                    )
        outputs = {}
        for name, tensors in per_task_outputs.items():
            out = tf.concat(tensors, axis=1)
            outputs[name] = out
        return outputs

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

    def get_build_training_targets_fn(self):
        self.check_built()

        def prepare_dataset_fn(batch_data: Dict[str, Any]):
            batch_data: ImageData[tf.Tensor] = ImageData.from_dict(batch_data)
            batch_frame = batch_data.labels
            labels = {}
            for fmt in self.fm_prediction_tasks:
                for name, target in fmt.get_targets(batch_frame).items():
                    labels[name] = target
            return batch_data.features.to_dict(), labels

        return prepare_dataset_fn

    def evaluate_model(
        self, model: keras.Model, eval_dataset: tf.data.Dataset, eval_steps: int
    ):
        optimizer = keras.optimizers.SGD()
        model.compile(optimizer, **self.get_model_compile_args())
        prepared_eval_dataset = eval_dataset.map(self.get_build_training_targets_fn())
        metrics = model.evaluate(prepared_eval_dataset, steps=eval_steps)
        return metrics

    def prepare_test_dataset(
        self, raw_dataset: tf.data.Dataset, batch_size: int, num_epochs: int = 1
    ) -> tf.data.Dataset:

        dataset = prepare_dataset(
            dataset=raw_dataset,
            augmentation_fn=None,
            model_image_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            num_epochs=num_epochs,
        )

        def prepare_features_fn(data):
            data: ImageData[tf.Tensor] = ImageData.from_dict(data)
            return data.features.to_dict(), {}

        return dataset.map(prepare_features_fn)

    def convert_to_tflite(
        self,
        model: Union[str, Path, keras.Model],
        save_path: str,
        export_batch_size: int = 1,
        raw_dataset: Optional[tf.data.Dataset] = None,
        num_dataset_samples: int = 64,
        num_test_steps: int = 1,
        postprocess_outputs: bool = True,
        merge_feature_maps: bool = False,
        verify_converted_models: bool = True,
        convert_quantized_model: bool = True,
    ) -> Tuple[keras.Model, List[Path]]:

        dataset = None
        if raw_dataset is not None:
            LOGGER.info("Preparing dataset for export of quantized model")
            dataset = self.prepare_test_dataset(raw_dataset, export_batch_size)

        if not isinstance(model, keras.Model):
            weights_path = str(model)
            LOGGER.info(f"Building model with batch_size = {export_batch_size}")
            model = self.build(batch_size=export_batch_size, is_training=False)
            model.load_weights(weights_path)

        export_model = self.postprocess_model(
            model,
            batch_size=export_batch_size,
            postprocess_outputs=postprocess_outputs,
            merge_feature_maps=merge_feature_maps,
            name="export",
        )

        output_tflite_models = tflite_ops.convert_model_to_tflite(
            model=export_model,
            save_path=save_path,
            dataset=dataset if convert_quantized_model else None,
            num_dataset_samples=num_dataset_samples,
        )

        if raw_dataset is not None and verify_converted_models:
            for tflite_model in output_tflite_models:
                self.verify_converted_model(
                    raw_dataset=raw_dataset,
                    tflite_model=tflite_model,
                    keras_model=export_model,
                    num_test_steps=num_test_steps,
                    batch_size=export_batch_size,
                )

        return export_model, output_tflite_models

    def verify_converted_model(
        self,
        keras_model: keras.Model,
        tflite_model: Path,
        raw_dataset: tf.data.Dataset,
        batch_size: int,
        num_test_steps: int,
    ) -> Dict[str, Dict[str, List[float]]]:
        LOGGER.info(f"Testing converted model: {tflite_model.name}")
        return tflite_ops.verify_converted_model_output_statistics(
            dataset=self.prepare_test_dataset(raw_dataset, batch_size),
            keras_model=keras_model,
            tflite_model=tflite_model,
            num_test_steps=num_test_steps,
        )

    def as_box_detector(
        self,
        weights: Union[str, Path],
        is_quantized: bool = False,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.35,
        scores_type: str = "objectness",
    ) -> box_detector.BoxDetector:

        weights = Path(weights)
        if weights.suffix in [".h5", ".hdf5"]:
            if is_quantized:
                model = self.build_quantized(batch_size=None, is_training=False)
            else:
                model = self.build(batch_size=None, is_training=False)

            try:
                model.load_weights(str(weights))
            except ValueError as e:
                LOGGER.error(
                    "Cannot load weights, try to build box detector "
                    f"with quantized=True if you are seeing 'axes don't match array' error. "
                )
                raise

            model = self.postprocess_model(
                model,
                batch_size=None,
                postprocess_outputs=True,
                merge_feature_maps=True,
                name="export",
            )
            detector_class = box_detector.FPNKerasBoxDetector
            output_names = model.output_names

        elif weights.suffix in [tflite_ops.TFLITE_SUFFIX]:
            LOGGER.info("Building box predictor for TFLite model")

            detector_class = box_detector.FPNTFLiteBoxDetector
            interpreter, predict_fn = tflite_ops.create_tflite_predict_fn(weights)
            output_details = interpreter.get_output_details()
            output_names = [o["name"] for o in output_details]

            def aug_predict_fn(image: np.ndarray):
                # TODO this should depend on input details
                assert image.shape[0] == 1, f"Expected batch size 1, got: {image.shape}"
                assert (
                    len(image.shape) == 4
                ), f"Expected input shape [1, H, W, 3], got {image.shape}"
                outputs = predict_fn([image])
                new_outputs = {}
                for n, array in outputs.items():
                    new_outputs[n.split("/")[1]] = array
                return new_outputs

            model = aug_predict_fn

        else:
            raise ValueError(
                f"Cannot infer model type from weights path: {weights}. "
                f"Use h5 or hdf5 extension for Keras model or .tflite "
                f"for TFLite model."
            )

        for name in output_names:
            if "fm" in name:
                raise ValueError(
                    "Cannot create box detector from model, since feature "
                    "maps are not merged. Please set merge_feature to True"
                )

        return detector_class(
            model,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            scores_type=scores_type,
        )


def reshape_feature_map(fm_out: tf.Tensor) -> tf.Tensor:
    fm_shape = fm_out.shape.as_list()
    if len(fm_shape) == 4:
        bs, h, w, c = fm_out.shape.as_list()
        if bs is None:
            bs = tf.shape(fm_out)[0]
        target_shape = [bs, h * w, c]
    elif len(fm_shape) == 3:
        bs, h, w = fm_out.shape.as_list()
        if bs is None:
            bs = tf.shape(fm_out)[0]
        target_shape = [bs, h * w]
    else:
        raise NotImplementedError(
            f"Cannot merge feature maps. Unsupported feature map shape: {fm_shape}"
        )

    fm_out = tf.reshape(fm_out, target_shape)
    return fm_out
