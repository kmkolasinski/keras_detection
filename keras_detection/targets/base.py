from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
import tensorflow as tf

from keras_detection.api import OutputTensorType
from keras_detection.structures import FeatureMapDesc, LabelsFrame


POSITIVE_ANCHOR = 1
NEGATIVE_ANCHOR = 0
IGNORED_ANCHOR = -1


@dataclass
class FeatureMapPredictionTarget(ABC):
    @property
    @abstractmethod
    def num_outputs(self) -> int:
        pass

    @property
    @abstractmethod
    def output_tensor_type(self) -> OutputTensorType:
        pass

    @property
    def num_outputs_with_weights(self) -> int:
        return self.num_outputs + 1

    @property
    def weights_index(self) -> int:
        return self.num_outputs

    @property
    def frame_required_fields(self) -> Optional[List[str]]:
        return None

    def has_weights(self) -> bool:
        return self.num_outputs_with_weights > self.num_outputs

    def get_targets_tensors(
        self, fm_desc: FeatureMapDesc, frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:

        if self.frame_required_fields is not None:
            frame.check_columns(self.frame_required_fields)

        @tf.function
        def target_per_batch(fields: List[tf.Tensor]):
            batch_frame = LabelsFrame.from_names_and_values(
                frame.non_empty_names, fields
            )
            return self.compute_targets(fm_desc=fm_desc, batch_frame=batch_frame)

        return target_per_batch(frame.non_empty_values)

    def create_targets_map(self, fm_desc: FeatureMapDesc, batch_size: int) -> tf.Tensor:
        return tf.zeros(
            [
                batch_size,
                fm_desc.fm_height,
                fm_desc.fm_width,
                self.num_outputs_with_weights,
            ]
        )

    def set_targets_shape(self, targets, fm_desc: FeatureMapDesc, batch_size: int):
        targets.set_shape(
            [
                batch_size,
                fm_desc.fm_height,
                fm_desc.fm_width,
                self.num_outputs_with_weights,
            ]
        )
        return targets

    def map_numpy_batch_compute_fn(
        self, fm_desc: FeatureMapDesc, numpy_fn, boxes, *args
    ) -> tf.Tensor:
        batch_size = boxes.shape[0]
        targets = self.create_targets_map(fm_desc, batch_size)
        targets = tf.numpy_function(
            numpy_fn, inp=[targets, boxes, *args], Tout=tf.float32
        )
        self.set_targets_shape(targets, fm_desc, batch_size)
        return targets

    @abstractmethod
    def compute_targets(
        self, fm_desc: FeatureMapDesc, batch_frame: LabelsFrame[tf.Tensor]
    ) -> tf.Tensor:
        pass

    def postprocess_predictions(
        self, fm_desc: FeatureMapDesc, predictions: tf.Tensor
    ) -> Union[tf.Tensor, Dict[str, tf.Tensor]]:
        return predictions

    def to_targets_and_weights(
        self, target: tf.Tensor
    ) -> Tuple[tf.Tensor, Optional[tf.Tensor]]:
        y_true = target[..., : self.num_outputs]
        if self.has_weights():
            weights = target[..., self.num_outputs :]
        else:
            weights = None
        return y_true, weights

    def to_pos_neg_ignored_anchors(self, target: tf.Tensor) -> tf.Tensor:
        """Returns tensor of shape [H, W, N] which indicates which anchors are """
        _, H, W, _ = target.shape
        B = tf.shape(target)[0]
        return tf.ones([B, H, W, 1])
