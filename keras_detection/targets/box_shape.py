from dataclasses import dataclass
from typing import List

import numpy as np
import tensorflow as tf
from numba import jit

from keras_detection import FeatureMapPredictionTarget
from keras_detection.ops import np_frame_ops
from keras_detection.structures import FeatureMapDesc, LabelsFrame


@dataclass
class BoxSizeTarget(FeatureMapPredictionTarget):
    @property
    def num_outputs(self) -> int:
        return 2

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_box_size_targets,
            batch_frame.boxes,
            batch_frame.num_rows,
        )


@dataclass
class BoxOffsetTarget(FeatureMapPredictionTarget):
    @property
    def num_outputs(self) -> int:
        return 2

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_box_offset_targets,
            batch_frame.boxes,
            batch_frame.num_rows,
        )


@dataclass
class BoxShapeTarget(FeatureMapPredictionTarget):
    @property
    def num_outputs(self) -> int:
        return 4

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:

        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_box_shape_targets,
            batch_frame.boxes,
            batch_frame.num_rows,
        )

    def to_pos_neg_ignored_anchors(self, target: tf.Tensor) -> tf.Tensor:
        _, weights = self.to_targets_and_weights(target)
        # box shape has no negatives only positives and ignored
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
                fm_scale_map[0, i, j, 0] = fm_desc.fm_height
                fm_scale_map[0, i, j, 1] = fm_desc.fm_width
                fm_scale_map[0, i, j, 2] = fm_desc.fm_height
                fm_scale_map[0, i, j, 3] = fm_desc.fm_width

        predictions = (predictions + shift_map) / fm_scale_map
        return predictions


@dataclass
class MeanBoxSizeTarget(FeatureMapPredictionTarget):
    @property
    def num_outputs(self) -> int:
        return 2

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_mean_box_size_targets,
            batch_frame.boxes,
            batch_frame.num_rows,
        )


@jit(nopython=True)
def batch_compute_box_size_targets(
    targets: np.ndarray, boxes: np.ndarray, num_rows: np.ndarray
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_box_size_targets(targets[i, :, :, :], boxes[i, : num_rows[i], :])
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:])", nopython=True)
def compute_box_size_targets(hw_map: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    heights, widths = np_frame_ops.boxes_heights_widths(boxes)
    fm_height, fm_width = hw_map.shape[:2]
    for k, (y, x) in enumerate(zip(y_center, x_center)):
        yi = int(y * fm_height)
        xi = int(x * fm_width)
        hw_map[yi, xi, 0] = heights[k] * fm_height
        hw_map[yi, xi, 1] = widths[k] * fm_width
        hw_map[yi, xi, 2] = 1
    return hw_map


@jit(nopython=True)
def batch_compute_box_offset_targets(
    targets: np.ndarray, boxes: np.ndarray, num_rows: np.ndarray
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_box_offset_targets(targets[i, :, :, :], boxes[i, : num_rows[i], :])
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:])", nopython=True)
def compute_box_offset_targets(offset_map: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    fm_height, fm_width = offset_map.shape[:2]
    for k, (y, x) in enumerate(zip(y_center, x_center)):
        yi = int(y * fm_height)
        xi = int(x * fm_width)
        offset_map[yi, xi, 0] = y * fm_height - yi
        offset_map[yi, xi, 1] = x * fm_width - xi
        offset_map[yi, xi, 2] = 1
    return offset_map


@jit(nopython=False)
def batch_compute_box_shape_targets(
    targets: np.ndarray, boxes: np.ndarray, num_rows: np.ndarray
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_box_shape_targets(targets[i, :, :, :], boxes[i, : num_rows[i], :])
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:])", nopython=True)
def compute_box_shape_targets(
    box_shape_map: np.ndarray, boxes: np.ndarray
) -> np.ndarray:

    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    heights, widths = np_frame_ops.boxes_heights_widths(boxes)
    fm_height, fm_width = box_shape_map.shape[:2]

    for k, (y, x) in enumerate(zip(y_center, x_center)):
        yi = int(y * fm_height)
        xi = int(x * fm_width)
        box_shape_map[yi, xi, 0] = heights[k] * fm_height
        box_shape_map[yi, xi, 1] = widths[k] * fm_width
        box_shape_map[yi, xi, 2] = y * fm_height - yi
        box_shape_map[yi, xi, 3] = x * fm_width - xi
        box_shape_map[yi, xi, 4] = 1
    return box_shape_map


@jit(nopython=True)
def batch_compute_mean_box_size_targets(
    targets: np.ndarray, boxes: np.ndarray, num_rows: np.ndarray
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_mean_box_size_targets(targets[i, :, :, :], boxes[i, : num_rows[i], :])
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:])", nopython=True)
def compute_mean_box_size_targets(hw_map: np.ndarray, boxes: np.ndarray) -> np.ndarray:

    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    heights, widths = np_frame_ops.boxes_heights_widths(boxes)
    fm_height, fm_width = hw_map.shape[:2]

    for k, (y, x) in enumerate(zip(y_center, x_center)):
        yi = int(y * fm_height)
        xi = int(x * fm_width)

        hw_map[yi, xi, 0] += heights[k] * fm_height
        hw_map[yi, xi, 1] += widths[k] * fm_width
        hw_map[yi, xi, 2] += 1

    hw_map[:, :, :2] = hw_map[:, :, :2] / (hw_map[:, :, 2:] + 1e-6)
    # make weights in range (0, 1)
    hw_map[:, :, 2] = np.minimum(hw_map[:, :, 2], 1)

    return hw_map