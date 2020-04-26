from typing import Tuple, List
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import keras_detection.losses as losses
import keras_detection.targets.box_objectness as box_obj
import keras_detection.targets.box_shape as sh_target
import keras_detection.tasks as dt
from keras_detection import FPNBuilder, Backbone, FeatureMapDesc
from keras_detection import metrics as m
from keras_detection.heads import (
    ActivationHead,
    HeadFactory,
)
from keras_detection.tasks import PredictionTaskDef
from keras_detection.utils.dvs import *

keras = tf.keras
LOGGER = tf.get_logger()


class SizeEstimatorBackbone(Backbone):
    def __init__(
        self,
        base_backbone: keras.Model,
        input_shape: Tuple[int, int, int],
        num_scales: int = 1,
        output_size: Tuple[int, int] = (5, 5),
    ):
        super().__init__(base_backbone, input_shape)
        self.num_scales = num_scales
        self.output_size = output_size
        self.size_projection = keras.layers.Conv2D(
            2, (1, 1), activation="sigmoid", name="size"
        )
        self.weight_projection = keras.layers.Conv2D(1, (1, 1), name="weight")

    @property
    def num_fm_maps(self) -> int:
        return 2

    def get_backbone(self, quantized: bool) -> keras.Model:
        if not quantized:
            return self.backbone
        else:
            LOGGER.info(
                f"Running quantization for model backbone: {self.backbone.name}"
            )
            return tfmot.quantization.keras.quantize_model(self.backbone)

    def project(self, outputs: tf.Tensor, proj_layer: keras.layers.Layer) -> tf.Tensor:
        h = proj_layer(outputs)
        return tf.image.resize(
            h, self.output_size, method=tf.image.ResizeMethod.BILINEAR
        )

    def forward(
        self, inputs: tf.Tensor, is_training: bool = False, quantized: bool = False
    ) -> List[tf.Tensor]:

        backbone = self.get_backbone(quantized=quantized)
        scales_outputs = []
        weights_outputs = []

        with tf.name_scope("SizeEstimator"):
            for scale in range(self.num_scales):

                sf = 2 ** scale
                LOGGER.info(f"Processing scale: {inputs}")
                outputs = backbone(inputs)
                if not isinstance(outputs, tf.Tensor):
                    outputs = outputs[-1]

                scales_outputs.append(sf * self.project(outputs, self.size_projection))
                weights_outputs.append(self.project(outputs, self.weight_projection))

                if scale < self.num_scales - 1:
                    inputs = keras.layers.AveragePooling2D(padding="same")(inputs)

        sizes: (B, H, W, 2, S) = tf.stack(scales_outputs, -1)
        weights_logits: (B, H, W, 1, S) = tf.stack(weights_outputs, -1)

        weights = tf.nn.softmax(weights_logits, axis=-1)
        mean_sizes: (B, H, W, 2, S) = weights * sizes
        mean_sizes: (B, H, W, 2) = tf.reduce_sum(mean_sizes, axis=-1)

        mean_sizes = tf.identity(mean_sizes, name="mean_size")

        weights_logits = tf.reduce_sum(weights_logits, axis=-1)
        weights_logits: (B, H, W, 1) = tf.identity(
            weights_logits, name="weights_logits"
        )
        return [mean_sizes, weights_logits]


class BoxSizeEstimatorBuilder(FPNBuilder):
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        base_backbone: keras.Model,
        box_size_task: PredictionTaskDef,
        objectness_task: PredictionTaskDef,
        num_scales: int = 2,
        output_size: Tuple[int, int] = (5, 5),
    ):
        if base_backbone.input_shape[1:3] != (None, None):
            raise ValueError(
                f"Backbone input shape height and width must "
                f"be None, got: {base_backbone.input_shape}"
            )

        backbone = SizeEstimatorBackbone(
            base_backbone, input_shape, num_scales=num_scales, output_size=output_size
        )

        tasks = [box_size_task, objectness_task]
        super().__init__(backbone, tasks)

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
        for fm_desc, task in zip(self.fm_descs, self.tasks):
            fm_task = dt.PredictionTask.from_task_def(
                f"{fm_desc.fm_name}/{task.name}", task
            )
            fm_tasks = dt.FeatureMapPredictionTasks(fm_desc.fm_name, fm_desc, [fm_task])
            self.fm_prediction_tasks.append(fm_tasks)


def get_mean_box_size_task(
    name: str = "box_shape", loss_weight: float = 5.0
) -> PredictionTaskDef:

    target = sh_target.MeanBoxSizeTarget()
    box_size = PredictionTaskDef(
        name=name,
        loss_weight=loss_weight,
        target_builder=target,
        head_factory=HeadFactory(num_outputs=target.num_outputs, htype=ActivationHead),
        loss=losses.L1Loss(target),
        metrics=[],
    )
    return box_size


def get_objectness_task(
    name: str = "objectness",
    loss_weight: float = 1.0,
    label_smoothing: float = 0.0,
    smooth_only_positives: bool = True,
    score_threshold: float = 0.3,
    pos_weights: float = 1.0,
) -> PredictionTaskDef:

    target = box_obj.BoxCenterIgnoreMarginObjectnessTarget(pos_weights=pos_weights)

    objectness_task = PredictionTaskDef(
        name=name,
        loss_weight=loss_weight,
        target_builder=target,
        head_factory=HeadFactory(
            num_outputs=target.num_outputs, activation="sigmoid", htype=ActivationHead
        ),
        loss=losses.BCELoss(
            target,
            label_smoothing=label_smoothing,
            smooth_only_positives=smooth_only_positives,
        ),
        metrics=[
            m.ObjectnessPrecision(target, score_threshold=score_threshold),
            m.ObjectnessRecall(target, score_threshold=score_threshold),
            m.ObjectnessPositivesMeanScore(target),
            m.ObjectnessNegativesMeanScore(target),
        ],
    )
    return objectness_task
