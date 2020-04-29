from collections import OrderedDict
from typing import Tuple, Optional, TypeVar, Generic, Any, List, Iterable, Dict
import dataclasses as dc
import numpy as np
import tensorflow as tf
from PIL import Image
from dataclasses import dataclass
from imgaug import BoundingBoxesOnImage
from keras_detection.ops import np_frame_ops
from keras_detection.utils import plotting

keras = tf.keras
Tensor = TypeVar("Tensor", tf.Tensor, np.ndarray)


@dataclass(frozen=True)
class DataClass:
    def replace(self, **kwargs) -> "DataClass":
        return dc.replace(self, **kwargs)

    def to_dict(self) -> Dict:
        # TypeError: can't pickle _thread.RLock objects
        return {k: v for k, v in zip(self.non_empty_names, self.non_empty_values)}

    @property
    def fields(self) -> List[dc.Field]:
        return dc.fields(self)

    @property
    def names(self) -> List[str]:
        return [f.name for f in self.fields]

    @property
    def non_empty_names(self) -> List[str]:
        return [n for n in self.names if getattr(self, n) is not None]

    def get(self, name: str) -> Any:
        return getattr(self, name)

    @property
    def non_empty_values(self) -> List[Any]:
        return [getattr(self, n) for n in self.names if getattr(self, n) is not None]

    @classmethod
    def from_dict(cls, params: Dict) -> "DataClass":
        return cls(**params)

    @classmethod
    def from_names_and_values(
        cls, names: List[str], values: Iterable[Any]
    ) -> "DataClass":
        return cls(**{n: v for n, v in zip(names, values)})


@dataclass(frozen=True)
class FeatureMapDesc(DataClass):
    fm_height: int
    fm_width: int
    image_height: int
    image_width: int

    @property
    def fm_name(self) -> str:
        return f"fm{self.fm_height}x{self.fm_width}"

    @property
    def y_stride(self) -> float:
        return self.image_height / self.fm_height

    @property
    def x_stride(self) -> float:
        return self.image_width / self.fm_width

    @property
    def stride(self) -> Tuple[float, float]:
        return self.y_stride, self.x_stride


@dataclass(frozen=True)
class LabelsFrame(DataClass, Generic[Tensor]):
    boxes: Optional[Tensor] = None
    labels: Optional[Tensor] = None
    weights: Optional[Tensor] = None
    num_rows: Optional[Tensor] = None

    def has_column(self, name: str) -> bool:
        is_not_none = not (getattr(self, name, None) is None)
        return is_not_none

    @property
    def columns(self) -> List[str]:
        return self.non_empty_names

    @property
    def length(self) -> Optional[int]:
        return self.boxes.shape[0]

    def check_columns(self, names: List[str]) -> None:
        if not all([self.has_column(n) for n in names]):
            raise ValueError(
                "Check columns failed. One or more of the required columns"
                f" is None.\nRequired columns: {names}.\n"
                f"Frame not None fields: {self.non_empty_names}\n"
                f"Frame content: {self}"
            )

    def to_imgaug_boxes(self, image: np.ndarray) -> BoundingBoxesOnImage:
        boxes = self.boxes
        assert isinstance(boxes, np.ndarray), "Only numpy arrays are supported"
        sy, sx = image.shape[:2]
        boxes = np_frame_ops.boxes_scale(boxes, sx, sy)
        boxes = boxes[:, (1, 0, 3, 2)]
        return BoundingBoxesOnImage.from_xyxy_array(boxes, (sy, sx))

    @staticmethod
    def from_imgaug_boxes(boxes: BoundingBoxesOnImage, image: np.ndarray) -> np.ndarray:
        sx, sy = image.shape[1], image.shape[0]
        new_boxes = boxes.to_xyxy_array()[:, (1, 0, 3, 2)]
        return np_frame_ops.boxes_scale(new_boxes, 1 / sx, 1 / sy)

    @classmethod
    def dataset_dtypes(cls):
        return {
            "boxes": tf.float32,
            "labels": tf.int64,
            "weights": tf.float32,
        }

    @classmethod
    def dataset_shapes(cls):
        return {
            "boxes": tf.TensorShape([None, 4]),
            "labels": tf.TensorShape([None]),
            "weights": tf.TensorShape([None]),
        }


@dataclass(frozen=True)
class Features(DataClass, Generic[Tensor]):
    image: Tensor

    @classmethod
    def dataset_dtypes(cls):
        return {"image": tf.uint8}

    @classmethod
    def dataset_shapes(cls):
        return {"image": tf.TensorShape([None, None, 3])}


@dataclass(frozen=True)
class ImageData(DataClass, Generic[Tensor]):
    features: Features[Tensor]
    labels: Optional[LabelsFrame[Tensor]] = None

    def to_dict(self) -> Dict:
        result = {"features": self.features.to_dict()}
        if self.labels is not None:
            result["labels"] = self.labels.to_dict()
        return result

    @classmethod
    def from_dict(cls, params: Dict) -> "ImageData":
        features = Features.from_dict(params["features"])
        labels = params.get("labels", None)
        if labels is not None:
            labels = LabelsFrame.from_dict(labels)
        return cls(features=features, labels=labels)

    def replace_image(self, image: Tensor) -> "ImageData":
        return self.replace(features=self.features.replace(image=image))

    def replace_frame_field(self, field: str, value: Tensor) -> "ImageData":
        frame = self.labels.replace(**{field: value})
        return self.replace(labels=frame)

    def has_labels(self) -> bool:
        return self.labels is not None

    @classmethod
    def dataset_dtypes(cls):
        return {
            "features": Features.dataset_dtypes(),
            "labels": LabelsFrame.dataset_dtypes(),
        }

    @classmethod
    def dataset_shapes(cls):
        return {
            "features": Features.dataset_shapes(),
            "labels": LabelsFrame.dataset_shapes(),
        }


def get_padding_shapes(
    dataset: tf.data.Dataset,
    spatial_image_shape: Tuple[int, int],
    max_num_boxes: Optional[int] = None,
) -> Dict[str, Any]:
    """Returns shapes to pad dataset tensors to before batching.

    Args:
        dataset: tf.data.Dataset object of Tuple features nested structure
            and labels nested structure.
        max_num_boxes: Max number of groundtruth boxes needed to computes
            shapes for padding. If None maximum number of boxes in batch is
            used instead.
        spatial_image_shape: A list of two integers of the form [height, width]
            containing expected spatial shape of the image.

    Returns:
        A nested structure of features and labels  padding shapes for
        tensors in the dataset.

    Raises:
        ValueError if dataset.output_shapes is not a tuple of two components
    """

    # dataset output shape
    try:
        shapes = dataset.output_shapes
    except AttributeError:
        shapes = tf.compat.v1.data.get_output_shapes(dataset)

    if type(shapes) != dict:
        raise ValueError("Dataset output types must be a dict with ")

    if len(shapes) != 2:
        raise ValueError("Dataset output types must be a dict of length 2")

    if "features" not in shapes:
        raise ValueError("Dataset output must contain 'features' key")

    if "labels" not in shapes:
        raise ValueError("Dataset output must contain 'labels' key")

    features_paddings = get_features_padding_shapes(
        shapes["features"], spatial_image_shape=spatial_image_shape
    )

    labels_paddings = get_labels_padding_shapes(
        shapes["labels"], max_num_boxes=max_num_boxes,
    )

    return {"features": features_paddings, "labels": labels_paddings}


def get_features_padding_shapes(
    features: Dict[str, Any], spatial_image_shape: Tuple[int, int]
) -> dict:
    """Returns shapes to pad dataset tensors to before batching.

    Args:
        features: nested dictionary of FeatureFields names with features
            fields to be pad in the dataset.padded_batch function.
        spatial_image_shape: A list of two integers of the form [height, width]
            containing expected spatial shape of the image.

    Returns:
        A nested structure of features and labels  padding shapes for
        tensors in the dataset.

    """
    height, width = spatial_image_shape
    features_padding_shape = OrderedDict([("image", [height, width, 3])])
    return features_padding_shape


def get_labels_padding_shapes(
    frame: Dict[str, Any], max_num_boxes: Optional[int],
) -> dict:
    """Returns shapes to pad dataset tensors to before batching.

    Args:
        frame: nested dictionary of LabelsFields with label fields to be
            pad in the dataset.padded_batch function.
        max_num_boxes: Max number of groundtruth boxes needed to computes
            shapes for padding.

    Returns:
        A nested structure of features and labels  padding shapes for
        tensors in the dataset.
    """

    labels_padding_shape = {
        "boxes": [max_num_boxes, 4],
        "labels": [max_num_boxes],
        "weights": [max_num_boxes],
        "num_rows": [],
    }

    frame: LabelsFrame[tf.TensorShape] = LabelsFrame.from_dict(frame)
    paddings = OrderedDict()
    for name in frame.non_empty_names:
        paddings[name] = labels_padding_shape[name]
    return paddings
