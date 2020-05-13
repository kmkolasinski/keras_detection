from enum import Enum
from typing import List

from dataclasses import dataclass

from keras_detection.structures import DataClass


@dataclass(frozen=True)
class LabelDescription(DataClass):
    name: str
    uuid: str


@dataclass(frozen=True)
class ModelMetadata(DataClass):
    name: str
    task: str
    # list of length equal to num_classes or 1 when only objectness is returned
    labels: List[LabelDescription]
    # optional additional output interpretation information
    output_interpretation: str = "default"
    # encoded json with optional task parameters
    task_params: str = ""
    version: str = "default"


class TaskType(Enum):
    OBJECT_LOCALIZATION = "object_localization"
    OBJECT_DETECTION = "object_detection"
    OBJECT_CLASSIFICATION = "object_classification"


class OutputTensorType(Enum):
    # [:, (height, width, center_y, center_x)] float
    BOX_SHAPE = "box_shape"
    # [:, (height, width, anchor_center_y, anchor_center_x)] float
    BOX_SIZE = "box_size"
    # [:, (anchor_cell_height, anchor_cell_width, center_y, center_x)] float
    BOX_OFFSET = "box_offset"
    # [:] float
    OBJECTNESS = "objectness"
    # [:, num_classes] float
    CLASSES = "classes"


class InputTensorsType(Enum):
    # [1, height, width, 3] uint8
    IMAGE = "image"
