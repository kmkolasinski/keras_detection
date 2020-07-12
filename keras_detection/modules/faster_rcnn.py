from abc import abstractmethod
from typing import List, Union, Any

import tensorflow as tf
from tensorflow.python.keras import Input

from keras_detection.modules.backbones.fpn import FPN
from keras_detection.modules.heads.heads import SingleConvHead
from keras_detection.modules.heads.rpn import RPN
from keras_detection.modules.retinanet import (
    NeuralGraph,
    Node,
    BoxShapeLoss,
    NodeLoss,
    BoxObjectnessLoss,
    InputNode,
)
from keras_detection.targets.box_shape import BoxShapeTarget
from keras_detection.targets.box_objectness import (
    BoxCenterIgnoreMarginObjectnessTarget,
    BoxCenterObjectnessTarget,
)
from keras_detection.structures import FeatureMapDesc, ImageData, LabelsFrame
import keras_detection.losses as losses
from keras_detection.modules.backbones.resnet import ResNet, FeatureMapDescEstimator
import keras_detection.models.utils as kd_utils

keras = tf.keras


class RPNLoss(NodeLoss):
    def __init__(
        self,
        inputs: List[str],
        box_input: str = "rpn/raw_boxes",
        obj_input: str = "rpn/objectness",
    ):
        super().__init__([*inputs, box_input, obj_input], weight=1.0)
        self.box_loss = BoxShapeLoss(inputs=[*inputs, box_input])
        self.obj_loss = BoxObjectnessLoss(inputs=[*inputs, obj_input])
        self.loss_tracker = {
            "boxes": self.box_loss.loss_tracker,
            "objectness": self.obj_loss.loss_tracker,
        }

    def call(
        self,
        fm_desc: FeatureMapDesc,
        batch_frame: LabelsFrame,
        boxes: tf.Tensor,
        objectness: tf.Tensor,
        per_anchor_loss=False,
    ):
        box_loss = tf.constant(self.box_loss.weight) * self.box_loss.call(
            fm_desc, batch_frame, boxes, per_anchor_loss=per_anchor_loss
        )
        obj_loss = tf.constant(self.obj_loss.weight) * self.obj_loss.call(
            fm_desc, batch_frame, objectness, per_anchor_loss=per_anchor_loss
        )
        return {"boxes": box_loss, "objectness": obj_loss}

    def __call__(
        self,
        fm_desc: FeatureMapDesc,
        batch_frame: LabelsFrame,
        boxes: tf.Tensor,
        objectness: tf.Tensor,
        per_anchor_loss: bool = False,
        training: bool = True,
    ):
        return self.call(
            fm_desc, batch_frame, boxes, objectness, per_anchor_loss=per_anchor_loss
        )


class FasterRCNNGraph(NeuralGraph):
    def __init__(self):
        super().__init__()

        image_dim = 224
        backbone = ResNet(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1, 1),
            num_last_blocks=2,
        )

        fpn = FPN(
            input_shapes=backbone.get_output_shapes(), depth=64, num_first_blocks=1
        )

        self.add(
            InputNode("image", getter=lambda d: ImageData.from_dict(d).features.image)
        )
        self.add(InputNode("labels", getter=lambda d: ImageData.from_dict(d).labels))

        self.add(Node("backbone", inputs=["image"], net=backbone))
        self.add(
            Node(
                "backbone/fm0/desc",
                inputs=["image", "backbone/fm0"],
                net=FeatureMapDescEstimator(),
            )
        )
        self.add(
            Node(
                "fpn",
                inputs=backbone.get_output_names("backbone"),
                inputs_as_list=True,
                net=fpn,
            )
        )

        rpn = RPN()
        rpn_loss = RPNLoss(inputs=["backbone/fm0/desc", "labels"])

        self.add(
            Node("rpn", inputs=["backbone/fm0/desc", "fpn/fm0"], net=rpn, loss=rpn_loss)
        )
        # For sampling proposals
        self.add(
            Node(
                "rpn/loss",
                inputs=rpn_loss.inputs,
                net=rpn_loss,
                call_kwargs={"per_anchor_loss": True},
            )
        )
