import tensorflow as tf

from keras_detection.targets.feature_map_sampling import scores_to_gather_indices, sample_feature_map


class FeatureMapSamplingTest(tf.test.TestCase):
    def test_scores_to_gather_indices(self):
        scores = tf.random.uniform(shape=[32, 64,  64, 1])
        features = tf.random.uniform(shape=[32, 64,  64, 5])
        indices = scores_to_gather_indices(scores, 3)
        sampled_features = sample_feature_map(features, indices)
        sampled_scores = sample_feature_map(scores, indices)

        exp_features = tf.reshape(features, [32, -1, 5])

        self.assertAllClose(exp_features[0, indices[0, 0]], sampled_features[0, 0])
        self.assertAllClose(exp_features[31, indices[31, 0]], sampled_features[31, 0])
        self.assertAllClose(exp_features[31, indices[31, 1]], sampled_features[31, 1])

        self.assertTrue(tf.greater(sampled_scores[0, 0, 0], sampled_scores[0, 1, 0]))
        self.assertTrue(tf.greater(sampled_scores[0, 1, 0], sampled_scores[0, 2, 0]))