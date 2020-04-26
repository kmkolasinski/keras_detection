import tensorflow as tf
import keras_detection.utils.testing_utils as utils
from keras_detection import ImageData, LabelsFrame, FeatureMapDesc
from keras_detection.targets.box_classes import MulticlassTarget
from keras_detection.targets.box_objectness import BoxCenterObjectnessTarget, \
    SmoothBoxCenterObjectnessTarget, BoxCenterIgnoreMarginObjectnessTarget
from keras_detection.targets.box_shape import BoxSizeTarget, BoxOffsetTarget, \
    BoxShapeTarget, MeanBoxSizeTarget

utils.maybe_enable_eager_mode()


class TargetsTest(tf.test.TestCase):
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

    def test_objectness_target_builder(self):
        image_data = self.sample_batch()
        tb = BoxCenterObjectnessTarget(pos_weights=10)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        self.assertEqual(
            targets.shape, [self.bs, *self.fm_size, 2]
        )

    def test_objectness_target_builder2(self):
        tb = BoxCenterObjectnessTarget()
        for _ in range(3):
            image_data = self.sample_batch()
            frame = image_data.labels
            tb.get_targets_tensors(self.fm_desc, frame)

    def test_smooth_objectness_target_builder(self):
        image_data = self.sample_batch()
        tb = SmoothBoxCenterObjectnessTarget(pos_weights=10)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        self.assertEqual(
            targets.shape, [self.bs, *self.fm_size, 2]
        )

    def test_ignore_margin_objectness_target_builder(self):
        image_data = self.sample_batch()
        tb = BoxCenterIgnoreMarginObjectnessTarget(pos_weights=10)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        self.assertEqual(
            targets.shape, [self.bs, *self.fm_size, 2]
        )

    def test_box_size_target_builder(self):
        image_data = self.sample_batch()
        tb = BoxSizeTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        y_pred, weights = tb.to_targets_and_weights(targets)
        self.assertEqual(y_pred.shape, [self.bs, *self.fm_size, 2])
        self.assertEqual(weights.shape, [self.bs, *self.fm_size, 1])

    def test_box_offset_target_builder(self):
        image_data = self.sample_batch()
        tb = BoxOffsetTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)

        y_pred, weights = tb.to_targets_and_weights(targets)
        self.assertEqual(y_pred.shape, [self.bs, *self.fm_size, 2])
        self.assertEqual(weights.shape, [self.bs, *self.fm_size, 1])

    def test_box_shape_target_builder(self):
        image_data = self.sample_batch()
        tb = BoxShapeTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        y_pred, weights = tb.to_targets_and_weights(targets)
        self.assertEqual(y_pred.shape, [self.bs, *self.fm_size, 4])
        self.assertEqual(weights.shape, [self.bs, *self.fm_size, 1])

    def test_classes_target_builder(self):

        image_data = self.sample_batch()
        tb = MulticlassTarget(5, add_dustbin=True)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        y_pred, weights = tb.to_targets_and_weights(targets)
        self.assertEqual(y_pred.shape, [self.bs, *self.fm_size, 6])
        self.assertEqual(weights.shape, [self.bs, *self.fm_size, 1])

        tb = MulticlassTarget(5, add_dustbin=False)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        y_pred, weights = tb.to_targets_and_weights(targets)
        self.assertEqual(y_pred.shape, [self.bs, *self.fm_size, 5])
        self.assertEqual(weights.shape, [self.bs, *self.fm_size, 1])

        with self.assertRaises(ValueError):
            frame = image_data.labels.replace(labels=None)
            tb = MulticlassTarget(5, add_dustbin=True)
            tb.get_targets_tensors(self.fm_desc, frame)

    def test_box_shape_target_postprocessing(self):

        frame = LabelsFrame(
            boxes=tf.constant([[[0.5, 0.3, 0.6, 0.6]]]),
            num_rows=tf.constant([1])
        )

        fm_desc = FeatureMapDesc(
            fm_height=4,
            fm_width=3,
            image_height=12,
            image_width=6,
        )

        tb = BoxShapeTarget()
        targets = tb.get_targets_tensors(fm_desc, frame)
        y_true, weights = tb.to_targets_and_weights(targets)
        recon = tb.postprocess_predictions(fm_desc, y_true)
        self.assertAllClose(recon[0, 2, 1], [0.1, 0.3, 0.55, 0.45])
        self.assertAllClose(weights[0, 2, 1], [1.0])

    def test_mean_box_size_target_builder(self):
        image_data = self.sample_batch()
        tb = MeanBoxSizeTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        y_pred, weights = tb.to_targets_and_weights(targets)
        self.assertEqual(y_pred.shape, [self.bs, *self.fm_size, 2])
        self.assertEqual(weights.shape, [self.bs, *self.fm_size, 1])


