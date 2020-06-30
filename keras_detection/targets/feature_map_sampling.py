import tensorflow as tf

from keras_detection.utils.dvs import *

keras = tf.keras


def sample_feature_map(features: (B, H, W, C), indices: (B, S)) -> tf.Tensor:
    """
    Args:
        features:
        indices: int32 indices to gather from features

    Returns:
        sampled_features: tensor of shape [B, S, C]
    """
    batch_size, num_channels = features.shape[0], features.shape[-1]
    return tf.gather(
        tf.reshape(features, [batch_size, -1, num_channels]), indices, batch_dims=1
    )


def scores_to_gather_indices(batch_scores: (B, H, W), num_samples_per_batch: int) -> tf.Tensor:
    """

    Args:
        batch_scores: locations with higher scores are sampled first
        num_samples_per_batch:

    Returns:

    """
    batch_size = batch_scores.shape[0]
    batch_scores = tf.reshape(batch_scores, [batch_size, -1])
    indices = tf.argsort(batch_scores, axis=-1, direction="DESCENDING")
    return indices[:, :num_samples_per_batch]
