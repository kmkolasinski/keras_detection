from abc import abstractmethod
from typing import List, Union, Any

import tensorflow as tf
from tensorflow.python.keras import Input

from keras_detection.layers.box_matcher import BoxMatcherLayer
from keras_detection.layers.box_regression import BoxRegressionTargetsBuilder
from keras_detection.modules.backbones.fpn import FPN
from keras_detection.modules.heads.heads import SingleConvHead
from keras_detection.modules.heads.rpn import RPN
from keras_detection.modules.layers import (
    ROISamplingLayer,
    ROINMSSamplingLayer,
    SimpleConvHeadLayer,
)
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
        per_anchor_loss: bool = False,
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


class L1Loss(NodeLoss):
    def __init__(self, inputs: List[str]):
        super().__init__(inputs, weight=10.0)
        self.loss_tracker = keras.metrics.Mean(name="box_regression_loss")

    def call(
        self,
        regression_targets: tf.Tensor,
        regression_weights: tf.Tensor,
        predictions: tf.Tensor,
    ):
        w = tf.constant(self.weight)
        box_loss = tf.reduce_sum(tf.abs(regression_targets - predictions), axis=-1)
        box_loss = box_loss * regression_weights
        box_loss = tf.reduce_mean(tf.reduce_mean(box_loss, -1), -1)
        return w * box_loss


class FasterRCNNGraph(NeuralGraph):
    def __init__(self, num_classes: int):
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

        roi_sampling = ROINMSSamplingLayer(num_samples=64, crop_size=(7, 7))
        self.add(
            Node(
                "roi_sampler",
                inputs=["fpn/fm0", "rpn/proposals", "rpn/objectness"],
                net=roi_sampling,
            )
        )

        self.add(
            Node(
                "box_matcher",
                inputs=["labels", "roi_sampler/proposals"],
                net=BoxMatcherLayer(),
            )
        )

        self.add(
            Node(
                "box_regression",
                inputs=["labels", "roi_sampler/proposals", "box_matcher/match_indices"],
                net=BoxRegressionTargetsBuilder(),
            )
        )

        # self.add(
        #     Node(
        #         "rois/classes",
        #         inputs=["roi_sampler/rois"],
        #         net=SimpleConvHeadLayer(
        #             num_filters=128, num_outputs=num_classes, activation="sigmoid"
        #         ),
        #     )
        # )

        self.add(
            Node(
                "rois/boxes",
                inputs=["roi_sampler/rois"],
                net=SimpleConvHeadLayer(num_filters=128, num_outputs=4),
                loss=L1Loss(
                    inputs=[
                        "box_regression/targets",
                        "box_regression/weights",
                        "rois/boxes",
                    ]
                ),
            )
        )
