import json
from pathlib import Path
from typing import Optional, List, Dict, Any

import tensorflow as tf
from tflite_support import flatbuffers
from tflite_support import metadata as _metadata
from tflite_support import metadata_schema_py_generated as _metadata_fb

from keras_detection.api import (
    LabelDescription,
    ModelMetadata,
    OutputTensorType,
    TaskType,
)

LOGGER = tf.get_logger()
keras = tf.keras
Lambda = keras.layers.Lambda


def build_metadata(
    name: str,
    version: str,
    labels: List[LabelDescription],
    output_types: List[OutputTensorType],
    output_interpretation: str = "",
    task: Optional[str] = None,
    author: str = "",
    task_params: str = ""
) -> bytearray:
    """

    Args:
        name:
        version:
        labels:
        output_types:
        output_interpretation:
        task:
        author:
        task_params:

    Returns:

    """
    if task is None or task == "":
        # Auto infer detection or localization from targets
        types = set(output_types)
        if types == {
            OutputTensorType.BOX_SHAPE,
            OutputTensorType.OBJECTNESS,
            OutputTensorType.CLASSES,
        }:
            task = TaskType.OBJECT_DETECTION.value
        elif types == {OutputTensorType.BOX_SHAPE, OutputTensorType.OBJECTNESS}:
            task = TaskType.OBJECT_LOCALIZATION.value
        else:
            raise ValueError(
                f"Cannot infer task_type from output_types: {output_types}"
            )

    metadata = ModelMetadata(
        name=name,
        version=version,
        task=task,
        output_interpretation=output_interpretation,
        labels=labels,
        task_params=task_params,
    )

    model_meta = _metadata_fb.ModelMetadataT()
    model_meta.name = name
    model_meta.description = json.dumps(metadata.asdict)
    model_meta.version = version
    model_meta.author = author

    input_meta = _metadata_fb.TensorMetadataT()
    input_meta.name = "image"

    output_metas = []
    for t in output_types:
        output_meta = _metadata_fb.TensorMetadataT()
        output_meta.name = t.value
        output_metas.append(output_meta)

    subgraph = _metadata_fb.SubGraphMetadataT()
    subgraph.inputTensorMetadata = [input_meta]
    subgraph.outputTensorMetadata = output_metas
    model_meta.subgraphMetadata = [subgraph]

    b = flatbuffers.Builder(0)
    b.Finish(model_meta.Pack(b), _metadata.MetadataPopulator.METADATA_FILE_IDENTIFIER)
    metadata_buf = b.Output()
    return metadata_buf


def dump_metadata(model_path: Path, metadata_buf: bytearray):
    """
    Adds metadata to TFLite model
    Args:
        model_path:
        metadata_buf:

    Returns:

    """
    populator = _metadata.MetadataPopulator.with_model_file(str(model_path))
    populator.load_metadata_buffer(metadata_buf)
    populator.populate()


def load_metadata(model_file: Path) -> Dict[str, Any]:
    """
    Parse metadata from TFLite model and return them as dictionary
    Args:
        model_file:

    Returns:

    """
    displayer = _metadata.MetadataDisplayer.with_model_file(str(model_file))
    json_file = displayer.get_metadata_json()
    return json.loads(json_file)
