from dataclasses import dataclass
from typing import List, Dict, Union
import numpy as np
import tensorflow as tf
from numba import jit
from keras_detection import FeatureMapPredictionTarget
from keras_detection.ops import np_frame_ops
from keras_detection.structures import FeatureMapDesc, LabelsFrame


@dataclass
class MulticlassTarget(FeatureMapPredictionTarget):
    num_classes: int
    add_dustbin: bool = True

    @property
    def num_outputs(self) -> int:
        return self.num_classes + self.add_dustbin

    @property
    def dustbin_index(self) -> int:
        return self.num_classes

    @property
    def frame_required_fields(self) -> List[str]:
        return ["boxes", "labels"]

    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        return self.map_numpy_batch_compute_fn(
            fm_desc,
            batch_compute_multiclass_targets,
            batch_frame.boxes,
            batch_frame.weights,
            batch_frame.num_rows,
            batch_frame.labels,
            self.add_dustbin,
            self.dustbin_index,
            self.weights_index,
        )

    def to_pos_neg_ignored_anchors(self, target: tf.Tensor) -> tf.Tensor:
        y_true, weights = self.to_targets_and_weights(target)
        if self.add_dustbin:
            mask = 1 - y_true[:, :, :, -1]
            return mask
        return super().to_pos_neg_ignored_anchors(target)

    def postprocess_predictions(
        self, fm_desc: FeatureMapDesc, predictions: tf.Tensor
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        if self.add_dustbin:
            predictions = predictions[:, :, :, :-1]
        return predictions


@jit(nopython=True)
def batch_compute_multiclass_targets(
    targets: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    num_rows: np.ndarray,
    labels: np.ndarray,
    add_dustbin: bool,
    dustbin_index: int,
    weights_index: int,
) -> np.ndarray:

    batch_size = targets.shape[0]
    for i in range(batch_size):
        compute_multiclass_targets(
            targets[i, :, :, :],
            boxes[i, : num_rows[i], :],
            weights[i, : num_rows[i]],
            labels[i, : num_rows[i]],
            add_dustbin=add_dustbin,
            dustbin_index=dustbin_index,
            weights_index=weights_index,
        )
    return targets


@jit(
    "float32[:,:,:](float32[:,:,:], float32[:,:], float32[:], int64[:], boolean, int64, int64)",
    nopython=True,
)
def compute_multiclass_targets(
    classes_map: np.ndarray,
    boxes: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    add_dustbin: bool,
    dustbin_index: int,
    weights_index: int,
) -> np.ndarray:

    y_center, x_center = np_frame_ops.boxes_clipped_centers(boxes)
    fm_height, fm_width = classes_map.shape[:2]

    classes_map[:, :, weights_index] = 1
    if add_dustbin:
        classes_map[:, :, dustbin_index] = 1

    for y, x, w, label_index in zip(y_center, x_center, weights, labels):
        yi = int(y * fm_height)
        xi = int(x * fm_width)
        classes_map[yi, xi, label_index] = 1.0
        classes_map[yi, xi, weights_index] = w
        if add_dustbin:
            classes_map[yi, xi, dustbin_index] = 0.0
    return classes_map
