import shutil
import tempfile

import tensorflow as tf

import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.models.box_size_estimator as bse
import keras_detection.ops.tflite_ops as tflite_ops
import keras_detection.utils.testing_utils as utils
from keras_detection import FPNBuilder
from keras_detection import ImageData
from keras_detection.backbones import resnet
from keras_detection.models.faster_rcnn_builder import FasterRCNNBuilder


def aug_fn(image_data: ImageData) -> ImageData:
    image = tf.cast(image_data.features.image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=1.2)
    return image_data.replace_image(image)


class SizeEstimatorBackboneTest(tf.test.TestCase):

    def create_dataset(self):
        image_dim = 64
        raw_dataset = datasets_ops.from_numpy_generator(
            utils.create_fake_detection_dataset_generator(num_steps=100)
        )

        return datasets_ops.prepare_dataset(
            raw_dataset,
            model_image_size=(image_dim, image_dim),
            augmentation_fn=aug_fn,
            num_epochs=-1,
            batch_size=2,
            shuffle_buffer_size=1,
            prefetch_buffer_size=1,
        )

    def test_coder(self):

        dataset = self.create_dataset()
        batch_data = next(iter(dataset))
        batch_data: ImageData[tf.Tensor] = ImageData.from_dict(batch_data)
        boxes = batch_data.labels.boxes

        bshape = boxes.shape.as_list()

        rpn_boxes = 1.01 * boxes + tf.random.uniform([bshape[0], bshape[1], 1], minval=-0.05, maxval=0.05)

        targets = FasterRCNNBuilder.encode_box_targets(
            rpn_boxes=rpn_boxes, target_boxes=boxes, target_weights=tf.ones_like(boxes[..., 0])
        )

        predictions = FasterRCNNBuilder.decode_rcnn_box_predictions(
            rpn_boxes=rpn_boxes, rcnn_boxes=targets[..., :4]
        )

        predictions = tf.reshape(predictions, bshape)

        self.assertAllClose(boxes, predictions)

    def test_real_case(self):
        target_boxes = tf.constant([[0.85663325, 0.10511288, 0.99999994, 0.25377652]])

        bshape = target_boxes.shape.as_list()

        predicted_boxes = tf.constant([[0.86021453, 0.11470187, 0.991154, 0.24762806]])

        targets = FasterRCNNBuilder.encode_box_targets(
            rpn_boxes=predicted_boxes, target_boxes=target_boxes, target_weights=tf.ones_like(target_boxes[..., 0])
        )

        predictions = FasterRCNNBuilder.decode_rcnn_box_predictions(
            rpn_boxes=predicted_boxes, rcnn_boxes=targets[..., :4]
        )

        predictions = tf.reshape(predictions, bshape)

        self.assertAllClose(target_boxes, predictions)
