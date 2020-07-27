import tensorflow as tf

import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.datasets.random_rectangles as random_rects
from keras_detection import ImageData
from keras_detection.modular.layers.box_matcher import BoxMatcherLayer
import numpy as np


class BoxMatcherTest(tf.test.TestCase):
    def test_matcher(self):
        dataset = datasets_ops.from_numpy_generator(
            random_rects.create_random_rectangles_dataset_generator()
        )
        dataset = datasets_ops.prepare_dataset(
            dataset, model_image_size=(128, 128), batch_size=3
        )

        batch_data = ImageData.from_dict(next(iter(dataset)))
        matcher = BoxMatcherLayer()

        matches = matcher(batch_data.labels, batch_data.labels.boxes[:, :5])

        exp_matches = np.array([[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
        self.assertAllEqual(matches, exp_matches)(matches, exp_matches)

        matches = matcher(batch_data.labels, batch_data.labels.boxes[:, 2:5])

        exp_matches = np.array([[2, 3, 4], [2, 3, 4], [2, 3, 4]])
        self.assertAllEqual(matches, exp_matches)(matches, exp_matches)

        matches = matcher(batch_data.labels, batch_data.labels.boxes[:, 2:5] * 0)

        exp_matches = np.array([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]])
        self.assertAllEqual(matches, exp_matches)

