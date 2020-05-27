import keras_detection.losses as losses
from keras_detection.targets.vector_field2d_estimation import (
    VectorField2DFromBoxesSequencesTarget,
)
from keras_detection.tasks import PredictionTaskDef
from keras_detection.heads import (
    SingleConvHeadFactory,
    SingleConvHead,
    NoQuantizableSingleConvHead,
)


def get_vector_field_2d_from_boxes_sequence_task(
    name: str = "vector_field_2d",
    loss_weight: float = 10.0,
    num_filters: int = 64,
    overlap_threshold: float = 0.1,
    quantizable: bool = False,
) -> PredictionTaskDef:
    target = VectorField2DFromBoxesSequencesTarget(overlap_threshold)
    return PredictionTaskDef(
        name=name,
        loss_weight=loss_weight,
        target_builder=target,
        head_factory=SingleConvHeadFactory(
            num_outputs=target.num_outputs,
            num_filters=num_filters,
            activation=None,
            htype=SingleConvHead if quantizable else NoQuantizableSingleConvHead,
        ),
        loss=losses.L1Loss(target),
        metrics=[],
    )
