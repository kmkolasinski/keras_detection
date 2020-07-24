import tensorflow as tf

from keras_detection import LabelsFrame, FeatureMapDesc
from keras_detection.modules.heads.rpn import RPN
from keras_detection.modules.layers import ROINMSSamplingLayer


class ROIAlignTest(tf.test.TestCase):

    def test_sampling(self):
        num_samples = 2
        batch_size = 4
        roi_sampler = ROINMSSamplingLayer(num_samples=num_samples, crop_size=(7, 7))
        feature_map = tf.random.uniform([batch_size, 32, 32, 128])
        rpn = RPN()
        fm_desc = FeatureMapDesc(32, 32, 256, 256)

        rpn_predictions = rpn(fm_desc, feature_map)
        roi_predictions = roi_sampler.call(
            feature_map,
            rpn_predictions["proposals"], rpn_predictions["objectness"]
        )

        self.assertEqual(roi_predictions['rois'].shape, [batch_size, num_samples, 7, 7, 128])
        self.assertEqual(roi_predictions['scores'].shape, [batch_size, num_samples])
        self.assertEqual(roi_predictions['proposals'].shape, [batch_size, num_samples, 4])
        self.assertEqual(roi_predictions['indices'].shape, [batch_size, num_samples])

        for i in range(batch_size):
            for j in range(num_samples):
                index = roi_predictions['indices'][i, j]
                exp_sampled_box = tf.reshape(rpn_predictions['proposals'][i], [-1, 4])[index]
                sampled_box = roi_predictions['proposals'][i, j]
                self.assertAllClose(exp_sampled_box, sampled_box)
