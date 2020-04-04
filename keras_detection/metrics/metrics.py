from abc import ABC
from dataclasses import dataclass
from typing import Tuple
import tensorflow as tf
from keras_detection.metrics.base import FeatureMapPredictionTargetMetric
from keras_detection.ops.np_frame_ops import epsilon
from keras_detection.targets.box_classes import MulticlassTarget
from keras_detection.utils.dvs import *

keras = tf.keras


@dataclass(frozen=True)
class ScoreBasedMetric(FeatureMapPredictionTargetMetric, ABC):
    score_threshold: float = 0.5

    @property
    def name(self) -> str:
        score = f"{int(self.score_threshold*100)}pc"
        return f"{self.__class__.__name__}AT{score}"

    def prepare(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S)
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        fm_height, fm_width = y_pred.shape[1:3]
        y_pred: (B, H * W) = tf.reshape(y_pred, [-1, fm_height * fm_width])
        y_true: (B, H * W) = tf.reshape(y_true, [-1, fm_height * fm_width])
        return y_true, y_pred


@dataclass(frozen=True)
class ObjectnessPrecision(ScoreBasedMetric):
    def compute(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S), weights: (B, H, W, 1)
    ) -> tf.Tensor:

        y_pred = tf.cast(tf.greater(y_pred, self.score_threshold), tf.float32)
        y_true, y_pred = self.prepare(y_true, y_pred)
        num_matches = tf.reduce_sum(y_pred * y_true, -1)
        num_predictions = tf.reduce_sum(y_pred, -1) + epsilon
        # precision per image
        precision = num_matches / num_predictions

        # does not take images w/o targets into account
        has_targets = tf.cast(tf.greater(tf.reduce_sum(y_pred, -1), 0.0), tf.float32)
        num_images = tf.reduce_sum(has_targets) + epsilon
        return tf.reduce_sum(precision) / num_images


@dataclass(frozen=True)
class ObjectnessRecall(ScoreBasedMetric):
    def compute(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        y_pred = tf.cast(tf.greater(y_pred, self.score_threshold), tf.float32)
        y_true, y_pred = self.prepare(y_true, y_pred)
        num_matches = tf.reduce_sum(y_pred * y_true, -1)
        num_targets = tf.reduce_sum(y_true, -1) + 1e-6
        recall = num_matches / num_targets
        return tf.reduce_mean(recall)


@dataclass(frozen=True)
class ObjectnessPositivesMeanScore(FeatureMapPredictionTargetMetric):
    def prepare(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S)
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        fm_height, fm_width = y_pred.shape[1:3]
        y_pred: (B, H * W) = tf.reshape(y_pred, [-1, fm_height * fm_width])
        y_true: (B, H * W) = tf.reshape(y_true, [-1, fm_height * fm_width])
        return y_true, y_pred

    def compute(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        y_true, y_pred = self.prepare(y_true, y_pred)
        summed_scores = tf.reduce_sum(y_pred * y_true, -1)
        num_targets = tf.reduce_sum(y_true, -1) + 1e-6
        average_score = summed_scores / num_targets
        return tf.reduce_mean(average_score)


@dataclass(frozen=True)
class ObjectnessNegativesMeanScore(ObjectnessPositivesMeanScore):
    def compute(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        y_true, y_pred = self.prepare(y_true, y_pred)
        y_negatives: (B, H * W) = 1.0 - y_true
        summed_scores: (B,) = tf.reduce_sum(y_pred * y_negatives, -1)
        num_targets: (B,) = tf.reduce_sum(y_negatives, -1) + 1e-6
        average_score = summed_scores / num_targets
        return tf.reduce_mean(average_score)


@dataclass(frozen=True)
class MulticlassAccuracyMetric(FeatureMapPredictionTargetMetric):
    target_def: MulticlassTarget
    has_dustbin_class: bool = True

    def compute(
        self, y_true: (B, H, W, S), y_pred: (B, H, W, S), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        dustbin = None
        if self.has_dustbin_class:
            dustbin: (B, H, W) = y_true[..., -1]
            y_pred: (B, H, W, S - 1) = y_pred[..., :-1]
            y_true: (B, H, W, S - 1) = y_true[..., :-1]

        accuracy: (B, H, W) = keras.metrics.categorical_accuracy(y_true, y_pred)
        fm_height, fm_width = y_pred.shape[1:3]
        accuracy: (B, H * W) = tf.reshape(accuracy, [-1, fm_height * fm_width])

        mask = tf.ones_like(accuracy)
        if dustbin is not None:
            mask = 1.0 - tf.reshape(dustbin, [-1, fm_height * fm_width])
            accuracy = accuracy * mask

        num_tp: (B,) = tf.reduce_sum(accuracy, -1)
        num_targets: (B,) = tf.reduce_sum(mask, -1) + 1e-6
        accuracy = num_tp / num_targets
        return tf.reduce_mean(accuracy)
