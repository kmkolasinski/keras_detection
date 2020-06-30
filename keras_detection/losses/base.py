from abc import abstractmethod, ABC
import tensorflow as tf
from keras_detection.targets.base import FeatureMapPredictionTarget

keras = tf.keras


class FeatureMapPredictionTargetLoss(tf.keras.losses.Loss, ABC):
    def __init__(self, target_def: FeatureMapPredictionTarget, per_anchor_loss: bool = False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_def = target_def
        self.per_anchor_loss = per_anchor_loss

    @property
    def __name__(self) -> str:
        return self.__class__.__name__

    def compute_per_anchor_loss(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        raise NotImplementedError("Per anchor loss is not implemented")

    @abstractmethod
    def compute_loss(
        self, y_true: tf.Tensor, y_pred: tf.Tensor, weights: tf.Tensor
    ) -> tf.Tensor:
        pass

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true, weights = self.target_def.to_targets_and_weights(y_true)

        fm_height, fm_width = y_pred.shape[1:3]
        if weights is None:
            weights = tf.ones([1, fm_height, fm_width, 1])

        if self.per_anchor_loss:
            loss = self.compute_per_anchor_loss(y_true, y_pred, weights=weights)
            loss = tf.reshape(loss, [-1, fm_height, fm_width])
            return loss

        return self.compute_loss(y_true, y_pred, weights=weights)
