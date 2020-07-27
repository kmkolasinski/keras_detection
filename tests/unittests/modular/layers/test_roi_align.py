import tensorflow as tf
from keras_detection import LabelsFrame
from keras_detection.modules.layers import roi_align
from keras_detection.modules.layers.roi_align import ROIAlignLayer


class ROIAlignTest(tf.test.TestCase):
    test_frame = LabelsFrame(
        boxes=tf.constant(
            [
                [[0.5, 0.3, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0]],
                [[0.1, 0.1, 0.6, 0.6], [0.0, 0.0, 0.0, 0.0]],
                [[0.7, 0.1, 0.6, 0.6], [0.7, 0.2, 0.5, 0.5]],
                [[0.1, 0.1, 0.6, 0.6], [0.2, 0.2, 0.5, 0.5]],
            ]
        ),
        num_rows=tf.constant([1, 1, 0, 2]),
    )

    def test_batch_frame_to_boxes(self):

        boxes, box_indices = roi_align.batch_frame_to_boxes(
            self.test_frame.boxes, self.test_frame.num_rows
        )
        self.assertAllClose(
            boxes,
            [
                [0.5, 0.3, 0.6, 0.6],
                [0.1, 0.1, 0.6, 0.6],
                [0.1, 0.1, 0.6, 0.6],
                [0.2, 0.2, 0.5, 0.5],
            ],
        )
        self.assertAllEqual(box_indices, [0, 1, 3, 3])

    def test_tf_batch_frame_to_boxes(self):

        boxes, box_indices = roi_align.tf_batch_frame_to_boxes(
            self.test_frame.boxes,
        )
        self.assertAllClose(
            boxes,
            [
                [0.5, 0.3, 0.6, 0.6],
                [0.0, 0.0, 0.0, 0.0],
                [0.1, 0.1, 0.6, 0.6],
                [0.0, 0.0, 0.0, 0.0],
                [0.7, 0.1, 0.6, 0.6],
                [0.7, 0.2, 0.5, 0.5],
                [0.1, 0.1, 0.6, 0.6],
                [0.2, 0.2, 0.5, 0.5],
            ],
        )
        self.assertAllEqual(box_indices, [0, 0, 1, 1, 2, 2, 3, 3])

    def test_batch_frame_symbolic(self):

        frame = LabelsFrame(
            boxes=tf.keras.Input(shape=[32, 64, 4], name="boxes"),
            num_rows=tf.keras.Input(shape=[32], name="num_rows", dtype=tf.int32),
        )

        boxes, box_indices = roi_align.tf_batch_frame_to_boxes(
            frame.boxes
        )

    def test_roi_align_layer(self):
        layer = ROIAlignLayer((32, 32))

        images = [
            1 * tf.ones([1, 64, 64, 3]),
            2 * tf.ones([1, 64, 64, 3]),
            3 * tf.ones([1, 64, 64, 3]),
            4 * tf.ones([1, 64, 64, 3]),
        ]
        images = tf.concat(images, axis=0)

        outputs = layer.call((images, self.test_frame.boxes))
        self.assertEqual(outputs.shape.as_list(), [8, 32, 32, 3])
        self.assertAllClose(outputs[0], 1 * tf.ones([32, 32, 3]))
        self.assertAllClose(outputs[1], 1 * tf.ones([32, 32, 3]))
        self.assertAllClose(outputs[2], 2 * tf.ones([32, 32, 3]))
        self.assertAllClose(outputs[3], 2 * tf.ones([32, 32, 3]))

    def test_symbolic(self):

        feature_map = tf.keras.Input(shape=[64, 64, 5], batch_size=32)
        frame = LabelsFrame(
            boxes=tf.keras.Input(shape=[52, 4], batch_size=32, name="boxes"),
        )
        layer = ROIAlignLayer((16, 16))
        outputs = layer([feature_map, frame.boxes])
        self.assertEqual(outputs.shape.as_list(), [32 * 52, 16, 16, 5])
