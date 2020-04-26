import tensorflow as tf
from keras_detection.backbones import resnet
from keras_detection.ops import tflite_ops
import tensorflow_model_optimization as tfmot


class ConversionTest(tf.test.TestCase):
    def test_model_conversion(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1,),
            num_last_blocks=1,
        )

        model = backbone.backbone
        qmodel = tfmot.quantization.keras.quantize_model(model)

        tflite_ops.TFLiteModel.from_keras_model(model)

        tflite_ops.TFLiteModel.from_keras_model(
            qmodel, [tf.lite.Optimize.DEFAULT]
        )
        tflite_ops.TFLiteModel.from_keras_model(
            qmodel, [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
        )
