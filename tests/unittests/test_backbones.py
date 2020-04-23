import tensorflow as tf

import keras_detection.utils.testing_utils as utils
from keras_detection.backbones import resnet
from keras_detection.backbones import fpn

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


class DenseFPNBackboneTest(tf.test.TestCase):
    image_dim = 64

    def build_fpn(self, num_first_blocks: int):

        backbone = resnet.ResNetBackbone(
            input_shape=(self.image_dim, self.image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )
        return fpn.FPNBackbone(
            backbone, depth=32, num_first_blocks=num_first_blocks
        )

    def test_forward(self):
        fpn_backbone = self.build_fpn(1)
        inputs = tf.keras.layers.Input(shape=[self.image_dim, self.image_dim, 3])
        fpn_backbone.fpn_backbone.summary()
        self.assertEqual(
            fpn_backbone.fpn_backbone.output_shape,
            [(None, 16, 16, 32), (None, 8, 8, 32)],
        )

        feature_maps = fpn_backbone.forward(inputs)
        self.assertEqual(feature_maps[0].shape.as_list(), [None, 16, 16, 32])

        fpn_backbone = self.build_fpn(2)
        self.assertEqual(
            fpn_backbone.fpn_backbone.output_shape,
            [(None, 16, 16, 32), (None, 8, 8, 32)],
        )

        feature_maps = fpn_backbone.forward(inputs)
        self.assertEqual(feature_maps[0].shape.as_list(), [None, 16, 16, 32])
        self.assertEqual(feature_maps[1].shape.as_list(), [None, 8, 8, 32])

    def test_quantize(self):
        fpn_backbone = self.build_fpn(2)
        inputs = tf.keras.layers.Input(shape=[self.image_dim, self.image_dim, 3])
        feature_maps = fpn_backbone.forward(inputs, quantized=True)
        self.assertEqual(len(feature_maps), 2)
