import tensorflow as tf
from keras_detection import ImageData

from keras_detection.losses import losses
import keras_detection.utils.testing_utils as utils
from keras_detection.targets.box_objectness import BoxCenterObjectnessTarget
from keras_detection.targets.box_shape import BoxSizeTarget
from keras_detection.targets.box_classes import MulticlassTarget

utils.maybe_enable_eager_mode()


class LossesTest(tf.test.TestCase):

    def setUp(self):
        self.bs = 3
        dataset = utils.create_fake_detection_batched_dataset(
            image_size=(64, 48), batch_size=self.bs, num_steps=100, num_classes=10
        )
        self.dataset = iter(dataset)
        self.fm_desc = utils.create_fake_fm_desc()
        self.fm_size = (self.fm_desc.fm_height, self.fm_desc.fm_width)

    def sample_batch(self):
        image_data = ImageData.from_dict(next(self.dataset))
        return image_data

    def test_bce_loss(self):
        image_data = self.sample_batch()
        tb = BoxCenterObjectnessTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        predictions = utils.create_fake_objectness_map(targets[..., :-1])
        self.assertEqual(predictions.shape, [self.bs, *self.fm_size, 1])

        bce_loss = losses.BCELoss(target_def=tb)
        loss = bce_loss(y_true=targets, y_pred=predictions)
        self.assertEqual(loss.shape, [])

    def test_softmax_ce_loss(self):
        image_data = self.sample_batch()
        tb = MulticlassTarget(num_classes=10)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        predictions = utils.create_fake_classes_map(targets[..., :-1])
        self.assertEqual(predictions.shape, [self.bs, *self.fm_size, 11])
        sce_loss = losses.SoftmaxCELoss(target_def=tb, label_smoothing=0.01)
        loss = sce_loss(y_true=targets, y_pred=predictions)
        self.assertEqual(loss.shape, [])

    def test_l1_loss(self):

        image_data = self.sample_batch()
        tb = BoxSizeTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        predictions = utils.create_fake_boxes_map(targets[..., :-1])
        self.assertEqual(predictions.shape, [self.bs, *self.fm_size, 2])

        l1_loss = losses.L1Loss(target_def=tb)
        loss = l1_loss(y_true=targets, y_pred=predictions)
        self.assertEqual(loss.shape, [])

    def test_bce_focal_loss(self):
        image_data = self.sample_batch()
        tb = MulticlassTarget(num_classes=10)
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        predictions = utils.create_fake_objectness_map(targets[..., :-1])
        self.assertEqual(predictions.shape, [self.bs, *self.fm_size, 11])

        bce_loss = losses.BCEFocalLoss(target_def=tb)
        loss = bce_loss(y_true=targets, y_pred=predictions)
        self.assertEqual(loss.shape, [])

        # compare implementations
        bce_focal_loss = losses.BCEFocalLoss(
            target_def=tb, alpha=0.5, gamma=0.0, label_smoothing=0.01)
        focal_loss = bce_focal_loss(y_true=targets, y_pred=predictions)

        bce_loss = losses.BCELoss(target_def=tb, label_smoothing=0.01)
        loss = bce_loss(y_true=targets, y_pred=predictions)
        self.assertAllClose(focal_loss, loss / 2)
