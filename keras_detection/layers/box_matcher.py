from typing import Dict

import numpy as np
import tensorflow as tf
from numba import jit

from keras_detection import LabelsFrame
from keras_detection.ops import np_frame_ops


class BoxMatcherLayer:

    def __init__(self, iou_threshold: float = 0.35):
        self.iou_threshold = iou_threshold

    def __call__(
        self, targets_batch_frame: LabelsFrame, predicted_boxes: tf.Tensor, training: bool = True
    ) -> Dict[str, tf.Tensor]:

        batch_size, num_sample = predicted_boxes.shape.as_list()[:2]
        match_indices = tf.zeros([batch_size, num_sample], dtype=tf.int64)
        match_indices = tf.numpy_function(
            batch_match_boxes,
            inp=[
                predicted_boxes,
                targets_batch_frame.boxes,
                targets_batch_frame.num_rows,
                self.iou_threshold,
                match_indices,
            ],
            Tout=tf.int64,
        )
        match_indices.set_shape([batch_size, num_sample])
        return {"match_indices": match_indices}


@jit(nopython=True)
def batch_match_boxes(
    predicted_boxes: np.ndarray,
    target_boxes: np.ndarray,
    num_rows: np.ndarray,
    iou_threshold: float,
    match_indices: np.ndarray,
) -> np.ndarray:
    batch_size = predicted_boxes.shape[0]
    for i in range(batch_size):
        match_boxes(
            predicted_boxes[i],
            target_boxes[i, : num_rows[i]],
            match_indices[i],
            iou_threshold,
        )
    return match_indices


@jit(nopython=True)
def match_boxes(
    predicted_boxes: np.ndarray,
    target_boxes: np.ndarray,
    match_indices: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    num_predicted = predicted_boxes.shape[0]
    iou = np_frame_ops.iou(predicted_boxes, target_boxes)
    for k in range(num_predicted):
        target_index = np.argmax(iou[k])
        if iou[k, target_index] > iou_threshold:
            match_indices[k] = target_index
        else:
            match_indices[k] = -1
    return match_indices
