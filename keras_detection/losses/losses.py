import tensorflow as tf

from keras_detection.losses.base import FeatureMapPredictionTargetLoss
from keras_detection.targets.base import FeatureMapPredictionTarget
from keras_detection.targets.box_classes import MulticlassTarget
from keras_detection.utils.dvs import *

keras = tf.keras


class BCELoss(FeatureMapPredictionTargetLoss):
    def __init__(
        self,
        target_def: FeatureMapPredictionTarget,
        label_smoothing: float = 0.01,
        smooth_only_positives: bool = True,
        from_logits: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(target_def, *args, **kwargs)
        self.label_smoothing = label_smoothing
        self.smooth_only_positives = smooth_only_positives
        self.from_logits = from_logits

    def smooth_labels(self, y_true: (B, H, W, C)) -> tf.Tensor:
        ls = self.label_smoothing
        if ls > 0.0:
            if self.smooth_only_positives:
                y_true = y_true * (1.0 - ls)
            else:
                y_true = y_true * (1.0 - ls) + 0.5 * ls
        return y_true

    def compute_per_anchor_loss(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        fm_height, fm_width = y_pred.shape[1:3]
        y_true = self.smooth_labels(y_true)
        loss: (B, H, W, C) = keras.backend.binary_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        loss: (B, H, W) = tf.reduce_sum(loss, axis=-1)
        loss: (B, H * W) = tf.reshape(loss, [-1, fm_height * fm_width])
        weights: (B, H * W) = tf.reshape(weights, [-1, fm_height * fm_width])
        return loss * weights

    def compute_loss(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        loss: (B, H * W) = self.compute_per_anchor_loss(y_true, y_pred, weights)
        loss: B = tf.reduce_mean(loss, -1)
        return loss


class SoftmaxCELoss(BCELoss):
    def compute_per_anchor_loss(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        fm_height, fm_width = y_pred.shape[1:3]
        y_true = self.smooth_labels(y_true)
        loss: (B, H, W) = keras.backend.categorical_crossentropy(
            y_true, y_pred, from_logits=self.from_logits
        )
        loss: (B, H * W) = tf.reshape(loss, [-1, fm_height * fm_width])
        weights: (B, H * W) = tf.reshape(weights, [-1, fm_height * fm_width])
        return loss * weights


class L1Loss(FeatureMapPredictionTargetLoss):
    def compute_loss(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:

        fm_height, fm_width = y_pred.shape[1:3]
        diff = y_pred - y_true
        loss: (B, H, W, 1) = tf.reduce_sum(tf.abs(diff), -1, keepdims=True)
        loss: (B, H * W) = tf.reshape(loss * weights, [-1, fm_height * fm_width])
        loss: (B,) = tf.reduce_mean(loss, -1)
        return loss


class CenteredBoxesIOULoss(FeatureMapPredictionTargetLoss):

    def compute_loss(
        self, y_true: (B, H, W, 2), y_pred: (B, H, W, 2), weights: (B, H, W, 1)
    ) -> tf.Tensor:

        fm_height, fm_width = y_pred.shape[1:3]

        area_pred = y_pred[..., 0] * y_pred[..., 1]
        area_true = y_true[..., 0] * y_true[..., 1]

        max_area: (B, H, W) = tf.maximum(area_true, area_pred) + 1e-6

        iou = tf.minimum(area_true, area_pred) / max_area
        iou = tf.clip_by_value(iou, 1e-5, 1.0)
        log_loss_iou = - tf.math.log(iou) * tf.reshape(weights, [-1, fm_height, fm_width])

        loss: (B, H * W) = tf.reshape(log_loss_iou, [-1, fm_height * fm_width])
        loss: (B,) = tf.reduce_mean(loss, -1)
        return loss


class BCEFocalLoss(BCELoss):
    def __init__(
        self,
        target_def: MulticlassTarget,
        alpha: float = 0.25,
        gamma: float = 2.0,
        label_smoothing: float = 0.01,
        smooth_only_positives: bool = True,
        from_logits: bool = False,
        *args,
        **kwargs
    ):
        super().__init__(target_def, label_smoothing, from_logits, *args, **kwargs)
        assert (
            target_def.add_dustbin
        ), "Focal loss computation is based on a dustbin variable!"
        self.alpha = alpha
        self.gamma = gamma
        self.smooth_only_positives = smooth_only_positives

    @property
    def focal_loss_fn(self):
        return get_focal_loss_fn(
            alpha=self.alpha,
            gamma=self.gamma,
            label_smoothing=self.label_smoothing,
            smooth_only_positives=self.smooth_only_positives,
            from_logits=self.from_logits,
        )

    def compute_loss(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:

        fm_height, fm_width, num_classes = y_pred.shape[1:]
        N = fm_height * fm_width

        labels = tf.reshape(y_true, [-1, N, num_classes])
        y_pred = tf.reshape(y_pred, [-1, N, num_classes])

        # 0 for background, 1 for object
        anchor_state: (B, N) = 1 - labels[..., -1]
        focal_loss: (B, N, C) = self.focal_loss_fn(
            labels=labels, anchor_state=anchor_state, y_pred=y_pred
        )
        # mean loss per class
        focal_loss: (B, N) = tf.reduce_sum(focal_loss, -1)
        weights = tf.reshape(weights, [-1, N])
        loss: (B, N) = focal_loss * weights
        # mean loss per feature map size
        loss: B = tf.reduce_mean(loss, -1)
        # keras will compute mean loss per batch
        return loss


def get_focal_loss_fn(
    alpha: float = 0.25,
    gamma: float = 2.0,
    label_smoothing: float = 0.0,
    smooth_only_positives: bool = False,
    from_logits: bool = False,
):
    """
    Copyright 2017-2018 Fizyr (https://fizyr.com)

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
                http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Create a functor for computing the focal loss.
    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """

    def _focal(
        labels: tf.Tensor, anchor_state: tf.Tensor, y_pred: tf.Tensor
    ) -> tf.Tensor:
        """ Compute the focal loss given the target tensor and the predicted tensor.
        As defined in https://arxiv.org/abs/1708.02002
        Args
            labels: Tensor of target data from the generator with shape (B, N, num_classes).
            anchor_state: Tensor of shape shape (B, N) with anchors
                flags -1 for ignore, 0 for background, 1 for object.
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        Returns
            The focal loss per anchor of shape [B, N, 1]
        """

        classification = y_pred
        fm_loss_shape = tf.shape(y_pred)
        # filter out "ignore" anchors
        indices = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(
            keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor
        )
        focal_weight = tf.where(
            keras.backend.equal(labels, 1), 1 - classification, classification
        )
        focal_weight = alpha_factor * focal_weight ** gamma

        if label_smoothing > 0.0:
            if smooth_only_positives:
                labels = labels * (1.0 - label_smoothing)
            else:
                labels = labels * (1.0 - label_smoothing) + 0.5 * label_smoothing

        cls_loss = focal_weight * keras.backend.binary_crossentropy(
            labels, classification, from_logits=from_logits
        )
        # shape must be int64
        fm_loss_shape = tf.cast(fm_loss_shape, tf.int64)
        fm_cls_loss = tf.scatter_nd(indices, cls_loss, shape=fm_loss_shape)
        return fm_cls_loss

    return _focal
