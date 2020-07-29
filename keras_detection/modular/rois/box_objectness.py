from typing import Dict

import numpy as np
import tensorflow as tf
from numba import jit

from keras_detection import LabelsFrame
from keras_detection.modular.core import Module
from keras_detection.ops import np_frame_ops


class BoxObjectnessTargetsBuilder(Module):

    def call(
        self,
        targets_batch_frame: LabelsFrame,
        sampled_proposals: tf.Tensor,
        match_indices: tf.Tensor,
        training: bool = True,
    ) -> Dict[str, tf.Tensor]:

        batch_size, num_samples, _ = sampled_proposals.shape.as_list()
        # plus weight
        objectness_targets = tf.zeros([batch_size, num_samples, 2])
        objectness_targets = tf.numpy_function(
            batch_box_objectness,
            inp=[
                targets_batch_frame.boxes,
                targets_batch_frame.num_rows,
                sampled_proposals,
                match_indices,
                objectness_targets,
            ],
            Tout=tf.float32,
        )
        objectness_targets.set_shape([batch_size, num_samples, 2])
        return {"targets": objectness_targets[..., :-1], "weights": objectness_targets[..., -1]}


@jit(nopython=True)
def batch_box_objectness(
    target_boxes: np.ndarray,
    num_rows: np.ndarray,
    sampled_proposals: np.ndarray,
    match_indices: np.ndarray,
    objectness_targets: np.ndarray,
) -> np.ndarray:

    batch_size = sampled_proposals.shape[0]
    for i in range(batch_size):
        box_objectness(
            target_boxes[i, : num_rows[i]],
            sampled_proposals[i],
            match_indices[i],
            objectness_targets[i],
        )
    return objectness_targets


@jit(nopython=True)
def box_objectness(
    target_boxes: np.ndarray,
    sampled_proposals: np.ndarray,
    match_indices: np.ndarray,
    objectness_targets: np.ndarray,
) -> np.ndarray:
    num_samples = sampled_proposals.shape[0]

    for k in range(num_samples):
        objectness_targets[k, 1] = 1.0
        target_index = match_indices[k]
        if target_index > -1:
            objectness_targets[k, 0] = 1.0

    return objectness_targets
