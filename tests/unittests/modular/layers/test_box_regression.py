import tensorflow as tf

import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.datasets.random_rectangles as random_rects
from keras_detection import ImageData
from keras_detection.modules.layers import BoxMatcherLayer
import numpy as np

from keras_detection.modules.layers import BoxRegressionTargetsBuilder


class BoxRegressionTargetBuilderTest(tf.test.TestCase):
    def test_matcher(self):
        dataset = datasets_ops.from_numpy_generator(
            random_rects.create_random_rectangles_dataset_generator()
        )
        dataset = datasets_ops.prepare_dataset(
            dataset, model_image_size=(128, 128), batch_size=3
        )

        batch_data = ImageData.from_dict(next(iter(dataset)))
        matcher = BoxMatcherLayer()

        proposals = batch_data.labels.boxes[:, :5]

        matches = matcher(batch_data.labels, proposals)

        target_builder = BoxRegressionTargetsBuilder()

        box_targets = target_builder(batch_data.labels, proposals, matches["match_indices"])

        self.assertAllClose(box_targets['targets'], np.zeros([3, 5, 4]))
        self.assertAllClose(box_targets['weights'], np.ones([3, 5]))

        matches = matcher(batch_data.labels, proposals * 0)
        box_targets = target_builder(batch_data.labels, proposals, matches["match_indices"])
        self.assertAllClose(box_targets['targets'], np.zeros([3, 5, 4]))
        self.assertAllClose(box_targets['weights'], np.zeros([3, 5]))