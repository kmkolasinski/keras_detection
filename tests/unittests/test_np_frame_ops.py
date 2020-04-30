from keras_detection.ops import np_frame_ops
import numpy as np
import tensorflow as tf

from keras_detection.utils.testing_utils import create_fake_boxframe


class FrameOpsTest(tf.test.TestCase):
    def test_iou(self):

        boxes = create_fake_boxframe().boxes.numpy()
        iou_mat = np_frame_ops.iou(boxes, boxes)
        self.assertAllClose(iou_mat, np.diag([1.0, 1.0, 1.0]))

    def test_iou_matching(self):
        boxes = create_fake_boxframe().boxes.numpy()

        left_indices, right_indices = np_frame_ops.argmax_iou_matching(boxes, boxes)
        self.assertAllEqual(left_indices, right_indices)
        self.assertAllEqual(left_indices, np.array([0, 1, 2]))

        left_indices, right_indices = np_frame_ops.argmax_iou_matching(boxes[:0], boxes)
        self.assertAllEqual(left_indices, np.array([]))
        self.assertAllEqual(right_indices, np.array([-1, -1, -1]))

        left_indices, right_indices = np_frame_ops.argmax_iou_matching(boxes, boxes[:0])
        self.assertAllEqual(left_indices, np.array([-1, -1, -1]))
        self.assertAllEqual(right_indices, np.array([]))

        left_indices, right_indices = np_frame_ops.argmax_iou_matching(boxes[:0], boxes[:0])
        self.assertAllEqual(left_indices, np.array([]))
        self.assertAllEqual(right_indices, np.array([]))

        left_indices, right_indices = np_frame_ops.argmax_iou_matching(boxes, boxes * 0)
        self.assertAllEqual(left_indices, np.array([-1, -1, -1]))
        self.assertAllEqual(right_indices, np.array([-1, -1, -1]))

        left_indices, right_indices = np_frame_ops.argmax_iou_matching(boxes, boxes[:2])
        self.assertAllEqual(left_indices, np.array([0, 1, -1]))
        self.assertAllEqual(right_indices, np.array([0, 1]))


