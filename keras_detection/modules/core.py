import tensorflow as tf

keras = tf.keras


class Module(keras.Model):
    pass

"""
graph = Graph("Faster-RCNN")

graph.add_input_node(InputNode(name="image", shape=[image_dim, image_dim, 3]))
graph.add_input_node(InputNode(name="boxes"), shape=[None, 4])
graph.add_input_node(InputNode(name="labels"))

graph.add_node(
    Node(inputs=["image"], module=ImagePreprocessor(name="normalized_image"))
)

graph.add_mode(
    Node(inputs=["normalized_image"], module=ResNet(name="feature_extractor"))
)

graph.add_node(
    Node(inputs=["feature_extractor/fm0"], train_inputs=["boxes", "labels"]),
    module=RPN(name="fm0/rpn"),
)

graph.add_node(
    Node(
        inputs=["fm0/rpn/proposals", "fm0/rpn/objectness"],
        train_inputs=["fm0/rpn/proposals_loss", "fm0/rpn/objectness_loss"],
        module=ROISampler(name="fm0/roi_sampler"),
    )
)

graph.add_mode(
    Node(
        inputs=["feature_extractor/fm0", "fm0/roi_sampler/proposals"],
        module=ROIExtractor(name="fm0/roi_extractor"),
    )
)

graph.add_node(
    Node(
        inputs=["fm0/roi_extractor/crops", "fm0/roi_sampler/proposals"],
        module=ROIBoxShapeHead(
            name="fm0/roi/box_shape",
            loss=RPNBoxLoss(
                inputs=[
                    "feature_extractor/fm0/fm_desc",
                    "boxes",
                    "fm0/roi_sampler/proposals",
                    "fm0/roi_sampler/indices",
                ]
            ),
        ),
    )
)

graph.add_node(
    Node(
        inputs=["fm0/roi_extractor/crops"],
        module=ROIClassesHead(
            name="fm0/roi/classes",
            loss=ROIClassesLoss(
                inputs=[
                    "feature_extractor/fm0/fm_desc",
                    "classes",
                    "fm0/roi_sampler/proposals",
                    "fm0/roi_sampler/indices",
                ]
            ),
        ),
    )
)


graph = Graph("RetinaNet")

graph.add_input_node(InputNode(name="image", shape=[image_dim, image_dim, 3], test_input=True))
graph.add_input_node(InputNode(name="boxes", shape=[None, 4]))
graph.add_input_node(InputNode(name="classes"))
graph.add_input_node(InputNode(name="labels"))

graph.add_node(
    Node(inputs=["image"], module=ImagePreprocessor(name="normalized_image"))
)

graph.add_mode(
    Node(inputs=["normalized_image"], module=ResNet(name="feature_extractor"))
)

graph.add_mode(
    Node(inputs=["feature_extractor/fm0", "feature_extractor/fm1", "feature_extractor/fm2"],
         module=FPN(name="fpn"))
)

graph.add_mode(
    Node(
        inputs=["fpn/fm0"],
        module=BoxShapeHead(name="fpn/fm0/box_shape", loss=BoxLoss(inputs=["boxes"])),
    )
)
graph.add_mode(
    Node(
        inputs=["fpn/fm0"],
        module=ObjectnessHead(
            name="fpn/fm0/objectness", loss=BCELoss(inputs=["boxes", "weights"])
        ),
    )
)
graph.add_mode(
    Node(
        inputs=["fpn/fm0"],
        module=ClassesHead(
            name="fpn/fm0/classes", loss=ClassesLoss(inputs=["boxes", "labels", "weights"])
        ),
    )
)

train_model = graph.to_keras_model(training=True)

"""