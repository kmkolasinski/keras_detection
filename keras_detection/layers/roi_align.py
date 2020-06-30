from typing import Tuple, List

import numpy as np
import tensorflow as tf
from numba import jit

from keras_detection import LabelsFrame
from keras_detection.utils.dvs import *
keras = tf.keras


class ROIAlignLayer(keras.layers.Layer):
    """Simplified version of the ROI Align, since crop_and_resize will
     result incorrect sampling."""
    def __init__(self, crop_size: Tuple[int, int], **kwargs):
        super(ROIAlignLayer, self).__init__(**kwargs)
        self.crop_size = crop_size

    def call(
        self, inputs: Tuple[tf.Tensor, tf.Tensor], **kwargs
    ) -> tf.Tensor:
        images, boxes = inputs
        boxes, box_indices = tf_batch_frame_to_boxes(boxes)
        outputs = tf.image.crop_and_resize(images, boxes, box_indices, self.crop_size)
        return outputs


@jit(nopython=True)
def flatten_boxes(
    boxes: np.ndarray,
    num_rows: np.ndarray,
    output_boxes: np.ndarray,
    box_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:

    batch_size = boxes.shape[0]
    index = 0
    for batch_index in range(batch_size):
        for box_index in range(num_rows[batch_index]):
            output_boxes[index, :] = boxes[batch_index, box_index, :]
            box_indices[index] = batch_index
            index += 1

    return output_boxes, box_indices


@tf.function
def batch_frame_to_boxes(boxes: tf.Tensor, num_rows: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    num_boxes = tf.reduce_sum(num_rows)
    output_boxes = tf.zeros([num_boxes, 4])
    box_indices = tf.zeros([num_boxes], dtype=tf.int32)

    output_boxes, box_indices = tf.numpy_function(
        flatten_boxes,
        inp=[boxes, num_rows, output_boxes, box_indices],
        Tout=[tf.float32, tf.int32],
    )
    output_boxes = tf.reshape(output_boxes, [num_boxes, 4])
    box_indices = tf.reshape(box_indices, [num_boxes])

    return output_boxes, box_indices


def tf_batch_frame_to_boxes(boxes: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """

    Args:
        boxes: tensor of shape [batch_size, num_boxes, 4]

    Returns:
        boxes: tensor of shape [batch_size * num_boxes, 4]
        indices: [batch_size * num_boxes] tf.int32
    """

    boxes_shape = tf.shape(boxes)
    batch_size = boxes_shape[0]
    # [batch_size, 1]
    indices = tf.reshape(tf.range(batch_size), [batch_size, 1])
    indices = tf.tile(indices, [1, boxes_shape[1]])

    boxes = tf.reshape(boxes, [-1, 4])
    indices = tf.reshape(indices, [-1])

    return boxes, indices