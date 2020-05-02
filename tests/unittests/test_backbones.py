import shutil
import tempfile

import tensorflow as tf

import keras_detection.utils.testing_utils as utils
from keras_detection.backbones import resnet
from keras_detection.backbones import fpn
from keras_detection.datasets.datasets_ops import from_numpy_generator
from keras_detection.ops.tflite_ops import TFLiteModel, convert_model_to_tflite


# TODO Consider running tests with different policies
# from tensorflow.keras.mixed_precision import experimental as mixed_precision
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


class ResNetBackboneTest(tf.test.TestCase):
    def test_quantize_resnet_forward(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )

        inputs = tf.keras.layers.Input(shape=[image_dim, image_dim, 3])

        feature_maps = backbone.forward(inputs)
        quantized_feature_maps = backbone.forward(inputs, quantized=True)


class FPNBackboneTest(utils.BaseUnitTest):
    image_dim = 64

    def build_fpn(self, num_first_blocks: int):

        backbone = resnet.ResNetBackbone(
            input_shape=(self.image_dim, self.image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )
        return fpn.FPNBackbone(backbone, depth=32, num_first_blocks=num_first_blocks)

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

    def test_quantize_forward(self):
        fpn_backbone = self.build_fpn(2)
        inputs = tf.keras.layers.Input(shape=[self.image_dim, self.image_dim, 3])
        feature_maps = fpn_backbone.forward(inputs, quantized=True)
        self.assertEqual(len(feature_maps), 2)

    def test_quantize_save(self):
        raw_dataset = utils.create_backbone_fake_representative_dataset(
            input_shape=(1, self.image_dim, self.image_dim, 3)
        )

        fpn_backbone = self.build_fpn(1)
        tflite_model = TFLiteModel.from_keras_model(fpn_backbone.as_model())
        tflite_model.test_predict()
        tflite_model = TFLiteModel.from_keras_model(
            fpn_backbone.as_model(), dataset=raw_dataset, num_samples=2
        )
        tflite_model.test_predict()

    def test_quantize_save_multiple_fms(self):
        raw_dataset = utils.create_backbone_fake_representative_dataset(
            input_shape=(1, self.image_dim, self.image_dim, 3)
        )

        fpn_backbone = self.build_fpn(2)
        model = fpn_backbone.as_model()
        tflite_model = TFLiteModel.from_keras_model(model)
        tflite_model.test_predict()

        tflite_model = TFLiteModel.from_keras_model(
            model, dataset=raw_dataset, num_samples=2
        )
        tflite_model.test_predict()
