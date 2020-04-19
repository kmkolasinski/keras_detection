import shutil
import tempfile

import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.utils.testing_utils as utils
import tensorflow as tf
from keras_detection import FPNBuilder
from keras_detection import ImageData
from keras_detection.backbones import resnet
from keras_detection.backbones.simple_cnn import SimpleCNNBackbone
from keras_detection.tasks import standard_tasks

utils.maybe_enable_eager_mode()


class ResNetBackboneTest(tf.test.TestCase):

    def test_quantize_resnet(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )

        inputs = tf.keras.layers.Input(shape=[image_dim, image_dim, 3])

        feature_maps = backbone.forward(inputs)
        quantized_feature_maps = backbone.forward(inputs, quantized=True)

