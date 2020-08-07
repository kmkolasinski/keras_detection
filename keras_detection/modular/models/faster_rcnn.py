from typing import List, Optional, Any, Dict

import tensorflow as tf
from keras_detection.models.box_detector import BoxDetector, BoxDetectionOutput
import numpy as np
from keras_detection.modular.core import (
    NodeLoss,
    NeuralGraph,
    Node,
    ImageInputNode,
    LabelsInputNode,
    KerasGraph,
)
from keras_detection.modular.rois.box_classes import BoxClassesTargetsBuilder
from keras_detection.modular.rois.box_matcher import BoxMatcherLayer
from keras_detection.modular.rois.box_objectness import BoxObjectnessTargetsBuilder
from keras_detection.modular.rois.box_regression import BoxRegressionTargetsBuilder
from keras_detection.modular.backbones.fpn import FPN
from keras_detection.modular.heads.rpn import RPN
from keras_detection.modular.layers.core import ROINMSSamplingLayer, SimpleConvHeadLayer
from keras_detection.modular.models.retinanet import BoxShapeLoss, BoxObjectnessLoss

from keras_detection.structures import FeatureMapDesc, LabelsFrame
from keras_detection.modular.backbones.resnet import ResNet
from keras_detection.modular.backbones.core import FeatureMapDescEstimator

keras = tf.keras


class RPNLoss(NodeLoss):
    def __init__(
        self,
        inputs: List[str],
        box_input: str = "rpn/raw_boxes",
        obj_input: str = "rpn/objectness",
        weight: float = 1.0,
    ):
        super().__init__([*inputs, box_input, obj_input], weight=weight)
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
    def __init__(self, inputs: List[str], weight: float = 10.0):
        super().__init__(inputs, weight=weight)
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


class BCELoss(NodeLoss):
    def __init__(
        self, inputs: List[str], name: str = "box_objectness_loss", weight: float = 1.0
    ):
        super().__init__(inputs, weight=weight)
        self.loss_tracker = keras.metrics.Mean(name=name)

    def call(
        self,
        objectness_targets: tf.Tensor,
        objectness_weights: tf.Tensor,
        predictions: tf.Tensor,
    ):
        w = tf.constant(self.weight)
        obj_loss = tf.losses.binary_crossentropy(
            objectness_targets, predictions, label_smoothing=0.02
        )
        obj_loss = obj_loss * objectness_weights
        obj_loss = tf.reduce_mean(tf.reduce_mean(obj_loss, -1), -1)
        return w * obj_loss


class FasterRCNNGraph(NeuralGraph):
    def __init__(
        self,
        image_dim: int,
        num_classes: Optional[int] = None,
        training: bool = True,
        backbone_params: Dict[str, Any] = {},
        fpn_params: Dict[str, Any] = {},
        roi_sampling_params: Dict[str, Any] = {},
        heads_params: Dict[str, Any] = {},
    ):
        super().__init__()

        self.image_dim = image_dim
        self.num_classes = num_classes
        self.backbone_params = backbone_params
        self.fpn_params = fpn_params
        self.roi_sampling_params = roi_sampling_params
        self.heads_params = heads_params
        self.build_graph(self, training=training)

    def build_graph(self, graph: NeuralGraph, training: bool) -> NeuralGraph:

        backbone = ResNet(
            input_shape=(self.image_dim, self.image_dim, 3),
            units_per_block=self.backbone_params.get("units_per_block", (1, 1, 1)),
            num_last_blocks=self.backbone_params.get("num_last_blocks", 2),
            init_filters=self.backbone_params.get("init_filters", 64),
        )

        graph.add(ImageInputNode())
        if training:
            graph.add(LabelsInputNode())

        graph.add(Node("backbone", inputs=["image"], module=backbone))
        graph.add(
            Node(
                "backbone/fm0/desc",
                inputs=["image", "backbone/fm0"],
                module=FeatureMapDescEstimator(),
            )
        )
        graph.add(
            Node(
                "fpn",
                inputs=backbone.get_output_names("backbone"),
                inputs_as_list=True,
                module=FPN(
                    input_shapes=backbone.get_output_shapes(),
                    depth=self.fpn_params.get("depth", 64),
                    num_first_blocks=self.fpn_params.get("num_first_blocks", 1),
                ),
            )
        )

        graph.add(
            Node(
                "rpn",
                inputs=["backbone/fm0/desc", "fpn/fm0"],
                module=RPN(),
                loss=RPNLoss(inputs=["backbone/fm0/desc", "labels"]),
            )
        )

        graph.add(
            Node(
                "roi_sampler",
                inputs=["fpn/fm0", "rpn/proposals", "rpn/objectness"],
                module=ROINMSSamplingLayer(
                    num_samples=self.roi_sampling_params.get("num_samples", 64),
                    crop_size=self.roi_sampling_params.get("crop_size", (7, 7)),
                ),
            )
        )
        if training:
            graph.add(
                Node(
                    "box_matcher",
                    inputs=["labels", "roi_sampler/proposals"],
                    module=BoxMatcherLayer(),
                )
            )

            graph.add(
                Node(
                    "rois_box_regression",
                    inputs=[
                        "labels",
                        "roi_sampler/proposals",
                        "box_matcher/match_indices",
                    ],
                    module=BoxRegressionTargetsBuilder(),
                )
            )

            graph.add(
                Node(
                    "rois_box_objectness",
                    inputs=[
                        "labels",
                        "roi_sampler/proposals",
                        "box_matcher/match_indices",
                    ],
                    module=BoxObjectnessTargetsBuilder(),
                )
            )

            if self.num_classes:
                graph.add(
                    Node(
                        "rois_box_classes",
                        inputs=[
                            "labels",
                            "roi_sampler/proposals",
                            "box_matcher/match_indices",
                        ],
                        module=BoxClassesTargetsBuilder(self.num_classes),
                    )
                )

        graph.add(
            Node(
                "rois/objectness",
                inputs=["roi_sampler/rois"],
                module=SimpleConvHeadLayer(
                    num_filters=self.heads_params.get("num_filters", 128),
                    num_outputs=1,
                    activation="sigmoid",
                ),
                loss=BCELoss(
                    inputs=[
                        "rois_box_objectness/targets",
                        "rois_box_objectness/weights",
                        "rois/objectness",
                    ]
                ),
            )
        )

        graph.add(
            Node(
                "rois/boxes",
                inputs=["roi_sampler/rois"],
                module=SimpleConvHeadLayer(
                    num_filters=self.heads_params.get("num_filters", 128), num_outputs=4
                ),
                loss=L1Loss(
                    inputs=[
                        "rois_box_regression/targets",
                        "rois_box_regression/weights",
                        "rois/boxes",
                    ]
                ),
            )
        )
        if self.num_classes:
            graph.add(
                Node(
                    "rois/classes",
                    inputs=["roi_sampler/rois"],
                    module=SimpleConvHeadLayer(
                        # plus dustbin
                        num_filters=self.heads_params.get("num_filters", 128),
                        num_outputs=self.num_classes + 1,
                        activation="sigmoid",
                    ),
                    loss=BCELoss(
                        name="box_classes_loss",
                        inputs=[
                            "rois_box_classes/targets",
                            "rois_box_classes/weights",
                            "rois/classes",
                        ],
                    ),
                )
            )
        return graph

    def as_predictor(self, batch_size: int, weights: str) -> keras.Model:
        image = tf.keras.Input(
            shape=(self.image_dim, self.image_dim, 3),
            batch_size=batch_size,
            name="image",
        )
        model = KerasGraph(
            graph=self.build_graph(NeuralGraph(), training=False), name="FasterRCNN"
        )
        predictor = keras.Model(image, model.call({"features": {"image": image}}))
        model.load_weights(weights)
        return predictor


class FasterRCNNBoxDetector(BoxDetector):
    def __init__(
        self,
        model: keras.Model,
        use_rpn_predictions: bool = False,
        score_threshold: float = 0.5,
        iou_threshold: float = 0.35,
    ):
        super().__init__(score_threshold, iou_threshold)
        self.model = model
        self.use_rpn_predictions = use_rpn_predictions

    def _predict(self, images: np.ndarray, **kwargs) -> List[BoxDetectionOutput]:

        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)

        predictions = self.model.predict(images)
        outputs = []
        for i in range(images.shape[0]):

            boxes = predictions["roi_sampler/proposals"][i]
            regression_boxes = predictions["rois/boxes"][i]

            scores = predictions["roi_sampler/scores"][i].reshape([-1])
            rois_scores = predictions["rois/objectness"][i].reshape([-1])
            rois_classes = None
            if "rois/classes" in predictions:
                rois_classes = predictions["rois/classes"][i]

            if not self.use_rpn_predictions:
                boxes = boxes + regression_boxes
                scores = rois_scores

            output = BoxDetectionOutput.from_tf_boxes(
                boxes=boxes,
                scores=scores,
                labels=rois_classes.argmax(-1) if rois_classes is not None else None,
                classes_scores=rois_classes,
            )
            outputs.append(output)

        return outputs