from typing import Tuple

import tensorflow as tf

from keras_detection.modular.layers.roi_align import ROIAlignLayer
from keras_detection.modular.core import Module, TrainableModule
from keras_detection.targets import feature_map_sampling as fm_sampling

keras = tf.keras


class ROISamplingLayer(TrainableModule):
    def __init__(self, num_samples: int, crop_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.roi_align = ROIAlignLayer(crop_size=crop_size)

    def call(
        self,
        feature_map: tf.Tensor,
        proposals: tf.Tensor,
        rpn_box_loss: tf.Tensor,
        rpn_obj_loss: tf.Tensor,
        training: bool = True,
    ):
        rpn_loss_map = rpn_box_loss + rpn_obj_loss

        indices = fm_sampling.scores_to_gather_indices(rpn_loss_map, self.num_samples)
        sampled_boxes = fm_sampling.sample_feature_map(proposals, indices)
        indices = tf.stop_gradient(indices)
        sampled_boxes = tf.stop_gradient(sampled_boxes)

        crops = self.roi_align([feature_map, sampled_boxes])
        return {
            "rois": crops,
            "proposals": sampled_boxes,
            "indices": indices
        }


class ROINMSSamplingLayer(Module):
    def __init__(self, num_samples: int, crop_size: Tuple[int, int], **kwargs):
        super().__init__(**kwargs)
        self.crop_size = crop_size
        self.num_samples = num_samples
        self.roi_align = ROIAlignLayer(crop_size=crop_size)

    def call(
        self,
        feature_map: tf.Tensor,
        proposals: tf.Tensor,
        scores: tf.Tensor,
        training: bool = True,
    ):

        batch_size = feature_map.shape[0]
        fm_height = feature_map.shape[1]
        fm_width = feature_map.shape[2]
        num_channels = feature_map.shape[3]
        max_output_size = fm_height * fm_width
        sampled_boxes = []
        sampled_scores = []
        sampled_indices = []
        for i in range(batch_size):
            batch_boxes = tf.reshape(proposals[i, ...], [-1, 4])
            batch_scores = tf.reshape(scores[i, ...], [-1])
            selected_indices = tf.image.non_max_suppression(
                batch_boxes,
                batch_scores,
                max_output_size=max_output_size,
                iou_threshold=0.5,
                score_threshold=0.1,
            )

            random_indices = tf.random.shuffle(list(range(max_output_size)))
            selected_indices = tf.concat([selected_indices, random_indices], axis=0)
            selected_indices = selected_indices[:self.num_samples]

            selected_boxes = tf.gather(batch_boxes, selected_indices)
            selected_scores = tf.gather(batch_scores, selected_indices)
            selected_scores = tf.reshape(selected_scores, [-1])

            sampled_indices.append(selected_indices)
            sampled_boxes.append(selected_boxes)
            sampled_scores.append(selected_scores)

        sampled_boxes = tf.stack(sampled_boxes, axis=0)
        sampled_scores = tf.stack(sampled_scores, axis=0)
        sampled_indices = tf.stack(sampled_indices, axis=0)
        crops = self.roi_align([feature_map, sampled_boxes])
        crops = tf.reshape(crops, [batch_size, self.num_samples, *self.crop_size, num_channels])
        sampled_boxes = tf.reshape(sampled_boxes, [batch_size, self.num_samples, 4])
        sampled_scores = tf.reshape(sampled_scores, [batch_size, self.num_samples])
        sampled_indices = tf.reshape(sampled_indices, [batch_size, self.num_samples])

        return {
            "rois": crops,
            "scores": sampled_scores,
            "proposals": sampled_boxes,
            "indices": sampled_indices
        }


class SimpleConvHeadLayer(TrainableModule):

    def __init__(self, num_filters: int, num_outputs: int, activation: str = None, **kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.num_outputs = num_outputs
        self.head_model = keras.Sequential([
            keras.layers.Conv2D(num_filters, kernel_size=3, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.Conv2D(num_filters, kernel_size=3, padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.ReLU(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(num_filters),
            keras.layers.ReLU(),
            keras.layers.Dense(num_outputs, use_bias=False),
        ])

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:

        batch_size, num_samples, height, width, num_channels = inputs.shape.as_list()
        inputs = tf.reshape(inputs, [batch_size * num_samples, height, width, num_channels])
        outputs = keras.layers.Activation(self.activation)(self.head_model(inputs))
        return tf.reshape(outputs, [batch_size, num_samples, self.num_outputs])
