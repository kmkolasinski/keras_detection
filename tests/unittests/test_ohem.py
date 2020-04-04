import keras_detection.utils.testing_utils as utils
import tensorflow as tf
from keras_detection import ImageData
from keras_detection.losses import BCELoss
from keras_detection.losses import ohem
from keras_detection.targets.box_objectness import BoxCenterObjectnessTarget
import numpy as np

utils.maybe_enable_eager_mode()


class OhemLossTest(tf.test.TestCase):

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

    def test_ohem_sampling_python(self):

        loss = np.array([
            [0.0, 0.0, 1.0, 0.2, 0.3, 0.1],
            [0.0, 0.15, 2.0, 0.3, 0.2, 0.1],
            [0.0, 0.15, 0.5, 0.3, 0.2, 0.1],
            [0.0, 0.15, 0.5, 0.3, 0.2, 0.1],
        ])

        weights = np.array([
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        ])

        out_pos_indices, out_neg_indices, num_pos, num_neg = ohem.batch_ohem(
            loss, weights,
            neg_pos_ratio=3,
            min_neg_per_image=2,
            out_pos_indices=np.zeros([loss.shape[0] * loss.shape[1]], dtype=np.int64),
            out_neg_indices=np.zeros([loss.shape[0] * loss.shape[1]], dtype=np.int64),
        )

        exp_pos_loss = np.array([1.0, 2.0, 0.2, 0.1])
        exp_neg_loss = np.array([
            0.3, 0.2, 0.1, 0.3, 0.2, 0.15,
            0.5, 0.3, 0.5, 0.3, 0.15, 0.0
        ])

        sampled_pos_loss = np.reshape(loss, [-1])[out_pos_indices[:num_pos]]
        sampled_neg_loss = np.reshape(loss, [-1])[out_neg_indices[:num_neg]]

        self.assertAllClose(exp_pos_loss, sampled_pos_loss)
        self.assertAllClose(exp_neg_loss, sampled_neg_loss)
        self.assertEqual(num_pos, 4)
        self.assertEqual(num_neg, 12)

    def test_ohem_loss(self):
        image_data = self.sample_batch()
        tb = BoxCenterObjectnessTarget()
        targets = tb.get_targets_tensors(self.fm_desc, image_data.labels)
        predictions = utils.create_fake_objectness_map(targets[..., :-1])

        bce_loss = ohem.OHEMBCELoss(target_def=tb)
        ohem_loss = ohem.OHEMLoss(
            target_def=tb,
            loss=bce_loss,
            neg_pos_ratio=3,
            min_neg_per_image=16,
            norm_by_num_pos=True
        )

        loss = ohem_loss(y_true=targets, y_pred=predictions)
        self.assertEqual(loss.shape, [])
