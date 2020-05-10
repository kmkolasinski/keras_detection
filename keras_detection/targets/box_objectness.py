from dataclasses import dataclass
from typing import List
import numpy as np
import tensorflow as tf
from numba import jit
from keras_detection import FeatureMapPredictionTarget
from keras_detection.api import OutputTensorType
from keras_detection.ops import np_frame_ops
from keras_detection.structures import FeatureMapDesc, LabelsFrame


@dataclass
class BoxCenterObjectnessTarget(FeatureMapPredictionTarget):
    pos_weights: float = 1.0
    from_logits: bool = False
    depth: int = 1

    @property
    def num_outputs(self) -> int:
        return self.depth

    @property
    def output_tensor_type(self) -> OutputTensorType:
        return OutputTensorType.OBJECTNESS

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_objectness_targets,
            batch_frame.boxes,
            batch_frame.weights,
            batch_frame.num_rows,
            self.pos_weights,
            self.num_outputs
        )

    def postprocess_predictions(
        self, fm_desc: FeatureMapDesc, predictions: tf.Tensor
    ) -> tf.Tensor:
        if self.from_logits:
            predictions = tf.nn.sigmoid(predictions)

        if self.depth == 1:
            scores = tf.squeeze(predictions, axis=-1)
        else:
            scores = tf.reduce_max(predictions, axis=-1)
        return scores

    def to_pos_neg_ignored_anchors(self, target: tf.Tensor) -> tf.Tensor:
        y_true, _ = self.to_targets_and_weights(target)
        mask = tf.cast(tf.greater(tf.reduce_sum(y_true, -1), 0.0), tf.float32)
        return mask


@jit(nopython=True)
def batch_compute_objectness_targets(
    targets: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    num_rows: np.ndarray,
    pos_weights: float,
    num_outputs: int,
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_objectness_targets(
            targets[i, :, :, :],
            boxes[i, : num_rows[i], :],
            weights[i, : num_rows[i]],
            pos_weights, num_outputs
        )
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:], float32[:], float32, int32)", nopython=True)
def compute_objectness_targets(
    targets: np.ndarray, boxes: np.ndarray, weights: np.ndarray, pos_weights: float, num_outputs: int
) -> np.ndarray:

    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    fm_height, fm_width = targets.shape[:2]
    targets[:, :, num_outputs] = 1
    for y, x, w in zip(y_center, x_center, weights):
        yi = int(y * fm_height)
        xi = int(x * fm_width)
        targets[yi, xi, :num_outputs] = 1
        targets[yi, xi, num_outputs] = pos_weights * w
    return targets


@dataclass
class SmoothBoxCenterObjectnessTarget(BoxCenterObjectnessTarget):
    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_smooth_objectness_targets,
            batch_frame.boxes,
            batch_frame.weights,
            batch_frame.num_rows,
            self.pos_weights,
            self.num_outputs
        )


@jit(nopython=True)
def batch_compute_smooth_objectness_targets(
    targets: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    num_rows: np.ndarray,
    pos_weights: float,
    num_outputs: int
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_smooth_objectness_targets(
            targets[i, :, :, :],
            boxes[i, : num_rows[i], :],
            weights[i, : num_rows[i]],
            pos_weights, num_outputs
        )
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:], float32[:], float32, int32)", nopython=True)
def compute_smooth_objectness_targets(
    targets: np.ndarray, boxes: np.ndarray, weights: np.ndarray, pos_weights: float, num_outputs: int
) -> np.ndarray:

    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    fm_h, fm_w = targets.shape[:2]

    targets[:, :, num_outputs] = 1
    for y, x, w in zip(y_center, x_center, weights):
        yi = int(y * fm_h)
        xi = int(x * fm_w)
        for j in [yi - 1, yi, yi + 1]:
            for i in [xi - 1, xi, xi + 1]:
                y_offset = y * fm_h - j
                x_offset = x * fm_w - i
                if 0 <= j < fm_h and 0 <= i < fm_w:
                    delta = -0.5 * (y_offset ** 2 + x_offset ** 2)
                    targets[j, i, :num_outputs] = np.exp(delta)
                    targets[j, i, num_outputs] = pos_weights * w

    return targets


@dataclass
class BoxCenterIgnoreMarginObjectnessTarget(BoxCenterObjectnessTarget):
    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_ignore_margin_objectness_targets,
            batch_frame.boxes,
            batch_frame.weights,
            batch_frame.num_rows,
            self.pos_weights,
            self.num_outputs
        )

    def to_pos_neg_ignored_anchors(self, target: tf.Tensor) -> tf.Tensor:
        raise NotImplementedError("Not implemented")


@jit(nopython=True)
def batch_compute_ignore_margin_objectness_targets(
    targets: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    num_rows: np.ndarray,
    pos_weights: float,
    num_outputs: int
) -> np.ndarray:
    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_ignore_margin_objectness_targets(
            targets[i, :, :, :],
            boxes[i, : num_rows[i], :],
            weights[i, : num_rows[i]],
            pos_weights, num_outputs
        )
    return targets


@jit("float32[:,:,:](float32[:,:,:], float32[:,:], float32[:], float32, int32)", nopython=True)
def compute_ignore_margin_objectness_targets(
    targets: np.ndarray, boxes: np.ndarray, weights: np.ndarray, pos_weights: float, num_outputs: int
) -> np.ndarray:

    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    fm_h, fm_w = targets.shape[:2]
    targets[:, :, num_outputs] = 1

    for y, x, w in zip(y_center, x_center, weights):
        yi = int(y * fm_h)
        xi = int(x * fm_w)
        targets[yi, xi, :num_outputs] = 1
        targets[yi, xi, num_outputs] = pos_weights * w

    for y, x, w in zip(y_center, x_center, weights):
        yi = int(y * fm_h)
        xi = int(x * fm_w)
        for j in [yi - 1, yi, yi + 1]:
            for i in [xi - 1, xi, xi + 1]:
                if 0 <= j < fm_h and 0 <= i < fm_w:
                    if targets[j, i, 0] != 1:
                        targets[j, i, num_outputs] = 0

    return targets
