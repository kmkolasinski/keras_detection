import tensorflow as tf

import keras_detection.utils.testing_utils as utils
from keras_detection.heads import SingleConvHead

utils.maybe_enable_eager_mode()


class HeadsTest(tf.test.TestCase):
    def test_single_conv_head(self):

        head = SingleConvHead(num_outputs=4, output_name="boxes")
        inputs = utils.create_fake_input_map(1, 32, 24, head.num_outputs)
        outputs = head.forward(inputs, quantized=False)
        self.assertEqual(outputs.shape.as_list(), [1, 32, 24, 4])
        q_outputs = head.forward(inputs, quantized=True)
        self.assertEqual(q_outputs.shape.as_list(), [1, 32, 24, 4])
