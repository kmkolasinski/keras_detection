import tensorflow as tf

from keras_detection import ImageData
from keras_detection.metrics.metrics import ObjectnessPrecision, MulticlassAccuracyMetric
import keras_detection.utils.testing_utils as utils
from keras_detection.targets.box_classes import MulticlassTarget
from keras_detection.targets.box_objectness import BoxCenterObjectnessTarget

utils.maybe_enable_eager_mode()


class MetricsTest(tf.test.TestCase):

    def setUp(self):
        self.bs = 3
        dataset = utils.create_fake_detection_batched_dataset(
            image_size=(64, 48), batch_size=self.bs, num_steps=100
        )
        self.dataset = iter(dataset)
        self.fm_desc = utils.create_fake_fm_desc()
        self.fm_size = (self.fm_desc.fm_height, self.fm_desc.fm_width)

    def sample_batch(self):
        image_data = ImageData.from_dict(next(self.dataset))
        return image_data

    def test_objectness_precision(self):

        image_data = self.sample_batch()
        frame = image_data.labels

        tb = BoxCenterObjectnessTarget()
        targets = tb.get_targets_tensors(self.fm_desc, frame)

        precision_metric = ObjectnessPrecision(target_def=tb)
        metric = precision_metric.get_metric_fn("test")
        self.assertEqual(metric.name, f"test/{precision_metric.name}")
        value = metric(y_true=targets, y_pred=targets[..., :-1])
        self.assertAllClose(value, 1.0, atol=1e-5)
        self.assertEqual(value.shape, [])

        value = metric(y_true=targets, y_pred=1 - targets[..., :-1])
        # as a mean of (1 + 0) / 2
        self.assertAllClose(value, 0.5, atol=1e-5)

    def test_multiclass_accuracy(self):

        image_data = self.sample_batch()
        frame = image_data.labels
        tb = MulticlassTarget(num_classes=3)
        targets = tb.get_targets_tensors(self.fm_desc, frame)
        precision_metric = MulticlassAccuracyMetric(target_def=tb)
        metric = precision_metric.get_metric_fn()
        value = metric(y_true=targets, y_pred=targets[..., :-1])
        self.assertAllClose(value, 1.0)
        self.assertEqual(value.shape, [])
