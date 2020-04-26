import numpy as np
import tensorflow as tf
import tensorflow_model_optimization as tfmo

from keras_detection.backbones import resnet
from keras_detection.ops.tflite_ops import TFLiteModel
from keras_detection.utils import tflite_debugger
from keras_detection.utils.tflite_debugger import OutputDiff
import pandas as pd


class DebugKerasModelTest(tf.test.TestCase):
    def test_convert_to_debug_model(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1,),
            num_last_blocks=1,
        )

        model = backbone.backbone

        qaware_model = tfmo.quantization.keras.quantize_model(model)

        debug_model = tflite_debugger.convert_to_debug_model(model)
        debug_qaware_model = tflite_debugger.convert_to_debug_model(qaware_model)

        self.assertEqual(len(debug_model.outputs), len(debug_qaware_model.outputs))

    def test_convert_to_tflite_debug_model(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )

        model = backbone.backbone
        qaware_model = tfmo.quantization.keras.quantize_model(model)

        debug_keras_model = tflite_debugger.convert_to_debug_model(qaware_model)
        debug_tflite_model = TFLiteModel.from_keras_model(debug_keras_model)

        self.assertEqual(
            len(debug_keras_model.output_names), len(debug_tflite_model.output_names)
        )

        matches = tflite_debugger.match_debug_models_output_names(
            debug_keras_model, debug_tflite_model
        )
        self.assertEqual(len(matches), len(debug_keras_model.output_names))

        inputs = np.random.rand(*[1, image_dim, image_dim, 3])

        output_diffs = tflite_debugger.diff_quantiztion_outputs(
            inputs,
            keras_model=debug_keras_model,
            tflite_model=debug_tflite_model
        )
        self.assertEqual(len(output_diffs), len(debug_tflite_model.output_names))

        def representative_dataset():
            while True:
                inputs = np.random.rand(*[1, image_dim, image_dim, 3])
                yield inputs

        output_diffs = tflite_debugger.debug_models_quantization(
            representative_dataset(),
            keras_model=debug_keras_model,
            tflite_model=debug_tflite_model, max_samples=2
        )
        self.assertEqual(len(output_diffs), len(debug_tflite_model.output_names))
        self.assertEqual(len(output_diffs[0].metrics['mae']), 2)


class OutputDiffTest(tf.test.TestCase):
    def test_output_diff(self):

        diff = OutputDiff(
            left_name="a",
            right_name="b",
            shape=(4, 5),
            left_dtype=np.float32,
            right_dtype=np.float32,
            metrics={"mse": [0, 1, 2, 1]},
        )
        self.assertEqual(diff.as_flat_dict()["metric/mse"], 1.0)

        df = OutputDiff.to_df([diff, diff])
        df = pd.DataFrame(df)
        df = df.set_index("left_name")
        self.assertEqual(df.shape, (2, 5))

