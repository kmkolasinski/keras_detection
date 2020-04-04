from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import tensorflow as tf

from keras_detection.losses import BCELoss
from keras_detection.losses.base import FeatureMapPredictionTargetLoss
from keras_detection.targets.base import FeatureMapPredictionTarget
from keras_detection.utils.dvs import *
from numba import jit

keras = tf.keras


class PosNegOHEMLoss(FeatureMapPredictionTargetLoss, ABC):
    def compute_positives_mask(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        mask = tf.cast(tf.greater(tf.reduce_sum(y_true, -1), 0.0), tf.float32)
        return mask


class OHEMBCELoss(BCELoss, PosNegOHEMLoss):
    def compute_positives_mask(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ):
        mask = tf.cast(tf.greater(tf.reduce_sum(y_true, -1), 0.0), tf.float32)
        return mask


class OHEMLoss(FeatureMapPredictionTargetLoss):
    def __init__(
        self,
        target_def: FeatureMapPredictionTarget,
        loss: PosNegOHEMLoss,
        neg_pos_ratio: float = 3,
        min_neg_per_image: int = 16,
        norm_by_num_pos: bool = True,
        add_summaries_steps: int = 16,
        *args,
        **kwargs,
    ):
        super().__init__(target_def, *args, **kwargs)
        self.loss = loss
        self.neg_pos_ratio = neg_pos_ratio
        self.min_neg_per_image = min_neg_per_image
        self.norm_by_num_pos = norm_by_num_pos
        self.add_summaries_steps = add_summaries_steps
        self.step = tf.Variable(0, dtype=tf.int64)

    def compute_loss(
        self, y_true: (B, H, W, C), y_pred: (B, H, W, C), weights: (B, H, W, 1)
    ) -> tf.Tensor:
        fm_height, fm_width = y_pred.shape[1:3]

        loss: (B, H * W) = self.loss.compute_per_anchor_loss(y_true, y_pred, weights)
        mask: (B, H, W) = self.loss.compute_positives_mask(y_true, y_pred, weights)
        mask: (B, H * W) = tf.reshape(mask, [-1, fm_height * fm_width])
        return self.sample(loss, mask, fm_size=(fm_height, fm_width))

    def sample_ohem_indices(self, loss: tf.Tensor, mask: tf.Tensor):
        num_anchors = loss.shape[-1]
        batch_size = tf.shape(loss)[0]

        args = [
            self.neg_pos_ratio,
            self.min_neg_per_image,
            tf.zeros([batch_size * num_anchors], dtype=tf.int64),
            tf.zeros([batch_size * num_anchors], dtype=tf.int64),
        ]

        out_pos_indices, out_neg_indices, num_pos, num_neg = tf.numpy_function(
            batch_ohem,
            inp=[loss, mask, *args],
            Tout=[tf.int64, tf.int64, tf.int64, tf.int64],
        )
        out_pos_indices = tf.reshape(out_pos_indices, [-1])
        out_neg_indices = tf.reshape(out_neg_indices, [-1])

        num_pos = tf.reshape(num_pos, [])
        num_neg = tf.reshape(num_neg, [])

        out_pos_indices = out_pos_indices[:num_pos]
        out_neg_indices = out_neg_indices[:num_neg]
        return out_pos_indices, out_neg_indices, num_pos, num_neg

    def sample(self, loss: tf.Tensor, mask: tf.Tensor, fm_size: Tuple[int, int]):
        # mean loss per num anchors and batch size
        input_loss = tf.reduce_mean(tf.reduce_mean(loss, -1))

        out_pos_indices, out_neg_indices, num_pos, num_neg = self.sample_ohem_indices(
            loss, mask
        )
        batch_size = tf.cast(tf.shape(loss)[0], tf.float32)
        loss = tf.reshape(loss, [-1])
        pos_loss = tf.gather(loss, out_pos_indices)
        neg_loss = tf.gather(loss, out_neg_indices)

        ohem_loss = tf.reduce_sum(pos_loss) + tf.reduce_sum(neg_loss)
        if self.norm_by_num_pos:
            ohem_loss = ohem_loss / tf.maximum(tf.cast(num_pos, tf.float32), 1.0)
        ohem_loss = ohem_loss / batch_size

        if self.add_summaries_steps > 0:

            @tf.function
            def with_summaries_fn(**kwargs):
                # this will work with TF autograph
                if self.step % self.add_summaries_steps == 0:
                    _loss = self.update_summaries(**kwargs, step=self.step)
                else:
                    _loss = kwargs["ohem_loss"]
                self.step.assign_add(1)
                return _loss

            ohem_loss = with_summaries_fn(
                out_pos_indices=out_pos_indices,
                out_neg_indices=out_neg_indices,
                input_loss=input_loss,
                ohem_loss=ohem_loss,
                ohem_pos_loss=pos_loss,
                ohem_neg_loss=neg_loss,
                num_pos=num_pos,
                num_neg=num_neg,
                batch_size=batch_size,
                fm_size=fm_size,
            )
        return ohem_loss

    def update_summaries(
        self,
        out_pos_indices: tf.Tensor,
        out_neg_indices: tf.Tensor,
        input_loss: tf.Tensor,
        ohem_loss: tf.Tensor,
        ohem_pos_loss: tf.Tensor,
        ohem_neg_loss: tf.Tensor,
        num_pos: tf.Tensor,
        num_neg: tf.Tensor,
        batch_size: tf.Tensor,
        fm_size: Tuple[int, int],
        step: Union[tf.Tensor, tf.Variable] = None,
    ) -> tf.Tensor:
        name = self.loss.__name__

        tf.summary.scalar(f"OHEM/{name}/Loss", input_loss, step=step)
        tf.summary.scalar(
            f"OHEM/{name}/PosLoss", tf.reduce_sum(ohem_pos_loss), step=step
        )
        tf.summary.scalar(
            f"OHEM/{name}/NegLoss", tf.reduce_sum(ohem_neg_loss), step=step
        )
        tf.summary.scalar(f"OHEM/{name}/SampledLoss", ohem_loss, step=step)

        num_samples = tf.cast(num_pos + num_neg, tf.float32) / batch_size
        num_pos = tf.cast(num_pos, tf.float32) / batch_size
        num_neg = tf.cast(num_neg, tf.float32) / batch_size

        tf.summary.scalar(f"OHEM/{name}/AvgNumSamples", num_samples, step=step)
        tf.summary.scalar(f"OHEM/{name}/AvgNumPos", num_pos, step=step)
        tf.summary.scalar(f"OHEM/{name}/AvgNumNeg", num_neg, step=step)

        size = tf.cast(batch_size * fm_size[0] * fm_size[1], tf.int64)
        out_pos_indices = tf.reshape(out_pos_indices, [-1, 1])
        out_neg_indices = tf.reshape(out_neg_indices, [-1, 1])

        def prepare_loss_map(anchors_loss):
            anchors_loss = tf.reshape(anchors_loss, [batch_size, -1])
            max_loss = tf.reduce_max(anchors_loss, axis=-1, keepdims=True)
            min_loss = tf.reduce_min(anchors_loss, axis=-1, keepdims=True)
            anchors_loss = (anchors_loss - min_loss) / (max_loss - min_loss + 1e-6)
            return tf.reshape(anchors_loss, [batch_size, *fm_size, 1])

        pos_loss_anchors = tf.scatter_nd(out_pos_indices, ohem_pos_loss, shape=[size])
        pos_loss_anchors = prepare_loss_map(pos_loss_anchors)

        neg_loss_anchors = tf.scatter_nd(out_neg_indices, ohem_neg_loss, [size])
        neg_loss_anchors = prepare_loss_map(neg_loss_anchors)

        tf.summary.image(
            f"OHEM/{name}/SampledPosAnchors", pos_loss_anchors, step=step, max_outputs=1
        )
        tf.summary.image(
            f"OHEM/{name}/SampledNegAnchors", neg_loss_anchors, step=step, max_outputs=1
        )

        return ohem_loss


@jit(nopython=True)
def batch_ohem(
    loss: np.ndarray,
    mask: np.ndarray,
    neg_pos_ratio: float,
    min_neg_per_image: int,
    out_pos_indices: np.ndarray,
    out_neg_indices: np.ndarray,
):

    batch_size, num_anchors = loss.shape
    num_pos_samples = 0
    num_neg_samples = 0

    for bidx in range(batch_size):
        anchors_loss = loss[bidx]
        anchors_weights = mask[bidx]

        pos_anchors = np.where(anchors_weights > 0)[0]

        neg_anchors = np.where(anchors_weights == 0)[0]
        pos_indices = np.argsort(anchors_loss[pos_anchors])[::-1]
        neg_indices = np.argsort(anchors_loss[neg_anchors])[::-1]
        pos_indices = pos_anchors[pos_indices]
        neg_indices = neg_anchors[neg_indices]

        # sample indices
        num_pos = len(pos_indices)
        num_neg = np.maximum(num_pos * neg_pos_ratio, min_neg_per_image)
        num_neg = np.minimum(num_neg, len(neg_indices))

        pos_indices = pos_indices[:num_pos]
        neg_indices = neg_indices[:num_neg]

        for i in pos_indices:
            out_pos_indices[num_pos_samples] = i + bidx * num_anchors
            num_pos_samples += 1

        for i in neg_indices:
            out_neg_indices[num_neg_samples] = i + bidx * num_anchors
            num_neg_samples += 1

    return out_pos_indices, out_neg_indices, num_pos_samples, num_neg_samples
