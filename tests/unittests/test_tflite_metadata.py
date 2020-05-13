from pathlib import Path

from keras_detection.api import OutputTensorType, ModelMetadata, LabelDescription
from keras_detection.backbones import resnet
from keras_detection.ops import tflite_ops
from keras_detection.ops.tflite_metadata import (
    build_metadata,
    dump_metadata,
    load_metadata,
)
from keras_detection.utils.testing_utils import BaseUnitTest


class TFLiteMetadataOpsTest(BaseUnitTest):
    def test_build_metadata(self):

        buffer = build_metadata(
            output_types=[OutputTensorType.OBJECTNESS, OutputTensorType.BOX_SHAPE],
            **ModelMetadata("test-name", "", labels=[]).asdict
        )

        buffer = build_metadata(
            output_types=[
                OutputTensorType.OBJECTNESS,
                OutputTensorType.BOX_SHAPE,
                OutputTensorType.CLASSES,
            ],
            **ModelMetadata("test-name", "", labels=[]).asdict
        )

        with self.assertRaises(ValueError):
            buffer = build_metadata(
                output_types=[OutputTensorType.OBJECTNESS],
                **ModelMetadata("test-name", "", labels=[]).asdict
            )

        buffer = build_metadata(
            output_types=[OutputTensorType.OBJECTNESS],
            **ModelMetadata("test-name", "loc", labels=[]).asdict
        )

    def test_dump_metadata(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1,),
            num_last_blocks=1,
        )
        model = backbone.backbone
        model_path = Path(self.test_dir) / "model.tflite"
        tflite_ops.TFLiteModel.from_keras_model(model).dump(model_path)

        buffer = build_metadata(
            output_types=[
                OutputTensorType.OBJECTNESS,
                OutputTensorType.BOX_SHAPE,
                OutputTensorType.CLASSES,
            ], author="xd",
            **ModelMetadata(
                "test-name",
                "",
                labels=[
                    LabelDescription("box", "box_uuid"),
                    LabelDescription("other_box", "other_box_uuid"),
                ],
            ).asdict,
        )

        dump_metadata(model_path, buffer)

        metadata = load_metadata(model_path)

        exp_metadata = {
            "name": "test-name",
            "description": '{"name": "test-name", "task": "object_detection", "labels": [{"name": "box", "uuid": "box_uuid"}, {"name": "other_box", "uuid": "other_box_uuid"}], "output_interpretation": "default", "task_params": "", "version": "default"}',
            "version": "default",
            "subgraph_metadata": [
                {
                    "input_tensor_metadata": [{"name": "image"}],
                    "output_tensor_metadata": [
                        {"name": "objectness"},
                        {"name": "box_shape"},
                        {"name": "classes"},
                    ],
                }
            ],
            "author": "xd",
        }
        self.assertEqual(metadata, exp_metadata)

        buffer = build_metadata(
            output_types=[
                OutputTensorType.OBJECTNESS,
                OutputTensorType.BOX_SHAPE,
            ], author="xd",
            **ModelMetadata(
                "test-name",
                "",
                labels=[
                    LabelDescription("box", "box_uuid"),
                    LabelDescription("other_box", "other_box_uuid"),
                ],
            ).asdict,
        )

        dump_metadata(model_path, buffer)

        metadata = load_metadata(model_path)

        exp_metadata = {
            "name": "test-name",
            "description": '{"name": "test-name", "task": "object_localization", "labels": [{"name": "box", "uuid": "box_uuid"}, {"name": "other_box", "uuid": "other_box_uuid"}], "output_interpretation": "default", "task_params": "", "version": "default"}',
            "version": "default",
            "subgraph_metadata": [
                {
                    "input_tensor_metadata": [{"name": "image"}],
                    "output_tensor_metadata": [
                        {"name": "objectness"},
                        {"name": "box_shape"},
                    ],
                }
            ],
            "author": "xd",
        }
        self.assertEqual(metadata, exp_metadata)
