from typing import Type, Union

import tensorflow as tf

from keras_detection.heads import SingleConvHeadFactory
import keras_detection.losses as losses
from keras_detection import metrics as m
from keras_detection.tasks import PredictionTaskDef
from keras_detection.targets.box_classes import MulticlassTarget
import keras_detection.targets.box_objectness as box_obj
import keras_detection.targets.box_shape as sh_target

keras = tf.keras

STANDARD_BOX_SHAPE_TARGETS = {
    "box_size": sh_target.BoxSizeTarget(),
    "box_offset": sh_target.BoxOffsetTarget(),
    "box_shape": sh_target.BoxShapeTarget(),
}

STANDARD_BOX_OBJECTNESS_TARGET_TYPE = {
    "center": box_obj.BoxCenterObjectnessTarget,
    "center_ignore_margin": box_obj.BoxCenterIgnoreMarginObjectnessTarget,
    "center_smooth": box_obj.SmoothBoxCenterObjectnessTarget,
}


def get_objectness_task(
    name: str = "objectness",
    loss_weight: float = 1.0,
    label_smoothing: float = 0.01,
    smooth_only_positives: bool = True,
    score_threshold: float = 0.2,
    pos_weights: float = 5.0,
    num_filters: int = 64,
    from_logits: bool = False,
    obj_class: Union[Type, str] = box_obj.BoxCenterObjectnessTarget,
) -> PredictionTaskDef:

    if isinstance(obj_class, str):
        obj_class = STANDARD_BOX_OBJECTNESS_TARGET_TYPE[obj_class]

    target = obj_class(pos_weights=pos_weights, from_logits=from_logits)
    activation = None if from_logits else "sigmoid"

    objectness_task = PredictionTaskDef(
        name=name,
        loss_weight=loss_weight,
        target_builder=target,
        head_factory=SingleConvHeadFactory(
            num_outputs=target.num_outputs,
            num_filters=num_filters,
            activation=activation,
        ),
        loss=losses.BCELoss(
            target,
            label_smoothing=label_smoothing,
            smooth_only_positives=smooth_only_positives,
            from_logits=from_logits,
        ),
        metrics=[
            m.ObjectnessPrecision(target, score_threshold=score_threshold),
            m.ObjectnessRecall(target, score_threshold=score_threshold),
            m.ObjectnessPositivesMeanScore(target),
            m.ObjectnessNegativesMeanScore(target),
        ],
    )
    return objectness_task


def get_box_shape_task(
    name: str = "box_shape", loss_weight: float = 10.0, num_filters: int = 64,
) -> PredictionTaskDef:
    target = STANDARD_BOX_SHAPE_TARGETS[name]
    return PredictionTaskDef(
        name=name,
        loss_weight=loss_weight,
        target_builder=target,
        head_factory=SingleConvHeadFactory(
            num_outputs=target.num_outputs, num_filters=num_filters, activation=None,
        ),
        loss=losses.L1Loss(target),
        metrics=[],
    )


def get_multiclass_task(
    num_classes: int,
    name: str = "classes",
    activation: str = "sigmoid",
    loss_weight: float = 1.0,
    label_smoothing: float = 0.01,
    smooth_only_positives: bool = False,
    fl_alpha: float = 0.25,
    fl_gamma: float = 2.0,
    num_filters: int = 64,
) -> PredictionTaskDef:
    target = MulticlassTarget(num_classes=num_classes)
    if activation == "sigmoid":
        loss = losses.BCEFocalLoss(
            target,
            label_smoothing=label_smoothing,
            smooth_only_positives=smooth_only_positives,
            alpha=fl_alpha,
            gamma=fl_gamma,
        )
        if fl_gamma == 0:
            loss = losses.BCELoss(
                target,
                label_smoothing=label_smoothing,
                smooth_only_positives=smooth_only_positives,
            )
    elif activation == "softmax":
        if fl_gamma > 0:
            raise NotImplementedError(
                "Focal loss is not supported for softmax activation."
            )
        loss = losses.SoftmaxCELoss(
            target,
            label_smoothing=label_smoothing,
            smooth_only_positives=smooth_only_positives,
        )
    else:
        raise ValueError("Invalid activation type")

    return PredictionTaskDef(
        name=name,
        loss_weight=loss_weight,
        target_builder=target,
        head_factory=SingleConvHeadFactory(
            num_outputs=target.num_outputs,
            num_filters=num_filters,
            activation=activation,
        ),
        loss=loss,
        metrics=[m.MulticlassAccuracyMetric(target)],
    )
