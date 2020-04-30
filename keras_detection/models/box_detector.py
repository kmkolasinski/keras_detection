from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Any, Dict, Tuple, Optional

import tensorflow as tf
import numpy as np
from PIL.Image import Image

from keras_detection.ops import np_frame_ops
from keras_detection.structures import DataClass, LabelsFrame
import keras_detection.utils.plotting as plotting
import matplotlib.pyplot as plt
from matplotlib import patches


@dataclass(frozen=True)
class BoxDetectionOutput(DataClass):
    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    classes_scores: np.ndarray

    @property
    def heights(self) -> np.ndarray:
        return self.boxes[:, 0]

    @property
    def widths(self) -> np.ndarray:
        return self.boxes[:, 1]

    @property
    def y_centers(self) -> np.ndarray:
        return self.boxes[:, 2]

    @property
    def x_centers(self) -> np.ndarray:
        return self.boxes[:, 3]

    @property
    def y_min(self) -> np.ndarray:
        return self.y_centers - self.heights / 2

    @property
    def x_min(self) -> np.ndarray:
        return self.x_centers - self.widths / 2

    @property
    def y_max(self) -> np.ndarray:
        return self.y_centers + self.heights / 2

    @property
    def x_max(self) -> np.ndarray:
        return self.x_centers + self.widths / 2

    @property
    def as_tf_boxes(self) -> np.ndarray:
        return np.stack([self.y_min, self.x_min, self.y_max, self.x_max]).T

    @property
    def num_boxes(self) -> int:
        return self.scores.shape[0]

    @property
    def num_classes(self) -> int:
        return self.classes_scores.shape[-1]

    def gather(self, indices: np.ndarray) -> "BoxDetectionOutput":
        fields = {n: self.get(n)[indices] for n in self.non_empty_names}
        return BoxDetectionOutput(**fields)

    def to_labels_frame(self) -> LabelsFrame[np.ndarray]:
        return LabelsFrame(boxes=self.as_tf_boxes, labels=self.labels)

    def draw(
        self,
        image: np.ndarray,
        figsize: Tuple[int, int] = (6, 6),
        title: Optional[str] = None,
        fontsize: int = 10,
        linewidth: float = 2,
        fmt: str = "png",
    ) -> Image:
        frame = self.to_labels_frame()
        return plotting.draw_labels_frame_boxes(
            image=image,
            boxes=frame.boxes,
            labels=self.labels,
            num_classes=self.num_classes,
            figsize=figsize,
            title=title,
            fontsize=fontsize,
            linewidth=linewidth,
            fmt=fmt,
        )

    @staticmethod
    def from_tf_boxes(
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: Optional[np.ndarray] = None,
        classes_scores: Optional[np.ndarray] = None,
    ) -> "BoxDetectionOutput":
        y, c = np_frame_ops.boxes_centers(boxes)
        h, w = np_frame_ops.boxes_heights_widths(boxes)
        boxes = np.stack([h, w, y, c], axis=-1)
        return BoxDetectionOutput(
            boxes=boxes,
            labels=labels,
            scores=scores or np.ones(shape=labels.shape),
            classes_scores=classes_scores or np.ones(shape=labels.shape),
        )


class BoxDetector:
    def __init__(self, score_threshold: float = 0.5, iou_threshold: float = 0.35):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def predict(self, images: np.ndarray, **kwargs) -> List[BoxDetectionOutput]:
        raw_detections = self._predict(images, **kwargs)
        outputs = []
        for detection in raw_detections:
            indices = tf.image.non_max_suppression(
                detection.as_tf_boxes,
                detection.scores,
                detection.num_boxes,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
            ).numpy()
            outputs.append(detection.gather(indices))
        return outputs

    @abstractmethod
    def _predict(self, images: np.ndarray, **kwargs) -> List[BoxDetectionOutput]:
        pass


class FPNKerasBoxDetector(BoxDetector):
    def __init__(
        self,
        model: Any,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.35,
        scores_type: str = "objectness",
    ):
        super().__init__(score_threshold, iou_threshold)
        self.model = model
        self.scores_type = scores_type

    def _scores_selection(
        self, objectness: np.ndarray, classes: np.ndarray
    ) -> np.ndarray:
        if self.scores_type == "objectness":
            return objectness
        elif self.scores_type == "classes":
            return classes
        elif self.scores_type == "objectness_and_classes":
            return classes * objectness
        elif self.scores_type == "objectness_or_classes":
            return np.maximum(classes, objectness)

    def _model_predict(self, images):
        names = self.model.output_names
        predictions = self.model.predict(x=images)
        return {n: predictions[k] for k, n in enumerate(names)}

    def _predict(self, images: np.ndarray, **kwargs) -> List[BoxDetectionOutput]:

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)

        predictions = self._model_predict(images)
        outputs = []
        for i in range(images.shape[0]):
            boxes = predictions["box_shape"][i].reshape([-1, 4])
            num_boxes = boxes.shape[0]
            objectness_scores = predictions["objectness"][i].reshape([-1])
            if "classes" in predictions:
                classes_scores = predictions["classes"][i].reshape([num_boxes, -1])
                labels = classes_scores.argmax(-1)
                scores = self._scores_selection(
                    objectness_scores, classes_scores.max(-1)
                )
            else:
                labels = np.array([0] * num_boxes)
                scores = objectness_scores
                classes_scores = objectness_scores.reshape([num_boxes, 1])

            output = BoxDetectionOutput(
                boxes=boxes, scores=scores, labels=labels, classes_scores=classes_scores
            )
            outputs.append(output)

        return outputs


class FPNTFLiteBoxDetector(FPNKerasBoxDetector):
    def _model_predict(self, images: np.ndarray) -> Dict[str, np.ndarray]:
        batched_predictions = defaultdict(list)
        for im in images:
            predictions = self.model(np.expand_dims(im, 0))
            for name, array in predictions.items():
                batched_predictions[name].append(array)

        predictions = {}
        for name, arrays in batched_predictions.items():
            predictions[name] = np.concatenate(arrays, axis=0)

        return predictions
