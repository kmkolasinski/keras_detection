from typing import List

import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from numba import jit

from keras_detection import FeatureMapPredictionTarget
from keras_detection.api import OutputTensorType
from keras_detection.ops import np_frame_ops
from keras_detection.structures import FeatureMapDesc, LabelsFrame


@dataclass
class VectorField2DFromBoxesSequencesTarget(FeatureMapPredictionTarget):
    """
    Creates 2D vector field estimation targets (dy, dx, y, x) from
    sequence of overlapping boxes. Target vector field is normalized

    Args:
         overlap_threshold: a minimal overlap threshold between two boxes
    """

    overlap_threshold: float = 0.1

    @property
    def num_outputs(self) -> int:
        return 4

    @property
    def output_tensor_type(self) -> OutputTensorType:
        return OutputTensorType.VECTOR_FIELD_2D

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:

        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_overlapping_boxes_direction_targets,
            batch_frame.boxes,
            batch_frame.num_rows,
            self.overlap_threshold,
        )

    def to_pos_neg_ignored_anchors(self, target: tf.Tensor) -> tf.Tensor:
        _, weights = self.to_targets_and_weights(target)
        #  has no negatives only positives and ignored
        mask = 2 * tf.cast(tf.greater(weights, 0.0), tf.float32) - 1
        return tf.squeeze(mask, axis=-1)

    def postprocess_predictions(
        self, fm_desc: FeatureMapDesc, predictions: tf.Tensor
    ) -> tf.Tensor:

        shift_map = np.zeros([1, fm_desc.fm_height, fm_desc.fm_width, 4]).astype(
            np.float32
        )
        fm_scale_map = np.ones([1, fm_desc.fm_height, fm_desc.fm_width, 4]).astype(
            np.float32
        )

        for i in range(fm_desc.fm_height):
            for j in range(fm_desc.fm_width):
                shift_map[0, i, j, 2] = i
                shift_map[0, i, j, 3] = j
                fm_scale_map[0, i, j, 0] = 1.0
                fm_scale_map[0, i, j, 1] = 1.0
                fm_scale_map[0, i, j, 2] = fm_desc.fm_height
                fm_scale_map[0, i, j, 3] = fm_desc.fm_width

        predictions = (predictions + shift_map) / fm_scale_map
        return predictions


@jit(nopython=False)
def batch_compute_overlapping_boxes_direction_targets(
    targets: np.ndarray,
    boxes: np.ndarray,
    num_rows: np.ndarray,
    overlap_threshold: float,
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_overlapping_boxes_direction_targets(
            targets[i, :, :, :], boxes[i, : num_rows[i], :], overlap_threshold
        )
    return targets


@jit(nopython=True)
def compute_overlapping_boxes_direction_targets(
    box_shape_map: np.ndarray, boxes: np.ndarray, overlap_threshold: float
) -> np.ndarray:
    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    fm_height, fm_width = box_shape_map.shape[:2]
    iou = np_frame_ops.iou(boxes, boxes)

    for k, (y, x) in enumerate(zip(y_center, x_center)):
        yi = int(y * fm_height)
        xi = int(x * fm_width)
        iou[k, k] = 0
        nns = np.argmax(iou[k])
        dx = 0
        dy = 0
        if iou[k, nns] > overlap_threshold:
            dy = y_center[nns] - y
            dx = x_center[nns] - x
            length = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
            # TODO assumption of normalized field
            dy = dy / length
            dx = dx / length

        box_shape_map[yi, xi, 0] = dy * np.sign(dx)
        # TODO dx is assumed to be always positive number
        box_shape_map[yi, xi, 1] = dx * np.sign(dx)
        box_shape_map[yi, xi, 2] = y * fm_height - yi
        box_shape_map[yi, xi, 3] = x * fm_width - xi
        box_shape_map[yi, xi, 4] = 1
    return box_shape_map
