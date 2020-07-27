import tensorflow as tf

from keras_detection import FeatureMapDesc
from keras_detection.modular.core import Module


class FeatureMapDescEstimator(Module):
    def call(
        self, image: tf.Tensor, feature_map: tf.Tensor, **kwargs
    ) -> FeatureMapDesc:
        fm_desc = FeatureMapDesc(
            *feature_map.shape[1:3].as_list(), *image.shape[1:3].as_list()
        )
        return fm_desc
