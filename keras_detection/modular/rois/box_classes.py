from typing import Dict

import numpy as np
import tensorflow as tf
from numba import jit

from keras_detection import LabelsFrame
from keras_detection.modular.core import Module
from keras_detection.ops import np_frame_ops


class BoxClassesTargetsBuilder(Module):

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def call(
        self,
        targets_batch_frame: LabelsFrame,
        sampled_proposals: tf.Tensor,
        match_indices: tf.Tensor,
        training: bool = True,
    ) -> Dict[str, tf.Tensor]:

        batch_size, num_samples, _ = sampled_proposals.shape.as_list()
        # plus dustbin, plus weight
        classes_targets = tf.zeros([batch_size, num_samples, self.num_classes + 2])
        classes_targets = tf.numpy_function(
            batch_box_classes,
            inp=[
                targets_batch_frame.boxes,
                targets_batch_frame.num_rows,
                targets_batch_frame.labels,
                sampled_proposals,
                match_indices,
                classes_targets,
            ],
            Tout=tf.float32,
        )
        classes_targets.set_shape([batch_size, num_samples, self.num_classes + 2])
        return {
            "targets": classes_targets[..., :-1],
            "weights": classes_targets[..., -1],
        }


@jit(nopython=True)
def batch_box_classes(
    target_boxes: np.ndarray,
    num_rows: np.ndarray,
    target_labels: np.ndarray,
    sampled_proposals: np.ndarray,
    match_indices: np.ndarray,
    classes_targets: np.ndarray,
) -> np.ndarray:

    batch_size = sampled_proposals.shape[0]
    for i in range(batch_size):
        box_classes(
            target_boxes[i, : num_rows[i]],
            target_labels[i, : num_rows[i]],
            sampled_proposals[i],
            match_indices[i],
            classes_targets[i],
        )
    return classes_targets


@jit(nopython=True)
def box_classes(
    target_boxes: np.ndarray,
    target_labels: np.ndarray,
    sampled_proposals: np.ndarray,
    match_indices: np.ndarray,
    classes_targets: np.ndarray,
) -> np.ndarray:
    num_samples = sampled_proposals.shape[0]

    for k in range(num_samples):
        # every sample has the same weight
        classes_targets[k, -1] = 1.0
        target_index = match_indices[k]
        if target_index > -1:
            target_label = target_labels[target_index]
            classes_targets[k, target_label] = 1.0
        else:
            # dustbin class
            classes_targets[k, -2] = 1.0

    return classes_targets
