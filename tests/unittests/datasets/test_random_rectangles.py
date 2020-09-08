import numpy as np
import tensorflow as tf

import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.datasets.random_rectangles as random_rects


class RandomRectangleTest(tf.test.TestCase):
    def test_draw_rectangle(self):

        image = np.zeros([100, 100, 3], dtype=np.float32)
        rect = np.array([0, 0, 10, 10], dtype=np.int64)
        image = random_rects.draw_rectangle(
            image=image,
            rect=rect,
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            alpha=1,
        )
        rect = np.array([90, 90, 99, 99], dtype=np.int64)
        random_rects.draw_rectangle(
            image=image,
            rect=rect,
            color=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            alpha=1.0,
        )

    def test_clip_rect(self):

        rect = np.array([0, 0, 10, 10], dtype=np.int64)
        new_rect = random_rects.clip_rect(rect, 20, 30)
        self.assertAllEqual(new_rect, rect)

        rect = np.array([-10, -2, 21, 32], dtype=np.int64)
        new_rect = random_rects.clip_rect(rect, 20, 30)
        self.assertAllEqual(new_rect, np.array([0, 0, 20, 30]))

    def test_clip_rect_to_image(self):
        image = np.zeros([20, 30, 3], dtype=np.float32)
        rect = np.array([-10, -2, 21, 32], dtype=np.int64)
        new_rect = random_rects.clip_rect_to_image(rect, image)
        self.assertAllEqual(new_rect, np.array([0, 0, 19, 29]))

    def test_box_intersection(self):
        recta = np.array([-10, -2, 21, 32], dtype=np.int64)
        rectb = np.array([-10, -2, 21, 32], dtype=np.int64)
        new_rect = random_rects.box_intersection(recta, rectb)
        self.assertAllEqual(rectb, new_rect)

        rectb = np.array([-20, -4, 31, 42], dtype=np.int64)
        new_rect = random_rects.box_intersection(recta, rectb)
        self.assertAllEqual(recta, new_rect)

        rectb = np.array([0, 0, 10, 10], dtype=np.int64)
        new_rect = random_rects.box_intersection(recta, rectb)
        self.assertAllEqual(rectb, new_rect)

        rectb = np.array([100, 100, 200, 200], dtype=np.int64)
        new_rect = random_rects.box_intersection(recta, rectb)
        self.assertAllEqual(np.array([0, 0, 0, 0], dtype=np.int64), new_rect)

    def test_box_union(self):
        recta = np.array([0, 0, 40, 10], dtype=np.int64)
        rectb = np.array([5, -10, 15, 20], dtype=np.int64)

        union = random_rects.box_union(recta, rectb)

        import matplotlib.pyplot as plt

        plt.plot(*union.T, "o-")
        plt.plot(*union.mean(0))
        plt.show()

    def test_draw_t_shapes(self):
        image = np.zeros([100, 100, 3], dtype=np.float32)
        rect = np.array([10, 10, 30, 30], dtype=np.int64)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        for i in range(9):
            image, polygon_mask = random_rects.draw_t_shape(
                image=image, rect=rect, color=color, shape_type=0, alpha=0.5
            )

        rect = np.array([-10, -10, 30, 30], dtype=np.int64)
        for i in range(9):
            image, polygon_mask = random_rects.draw_t_shape(
                image=image, rect=rect, color=color, shape_type=0, alpha=0.5
            )

    def test_draw_random_t_shape(self):
        image = np.zeros([100, 100, 3], dtype=np.float32)
        color = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        new_image, bbox, polygon_mask, label = random_rects.draw_random_t_shape(
            image=image, color=color
        )
        self.assertEqual(image.shape, new_image.shape)
        self.assertEqual(image.dtype, new_image.dtype)
        self.assertLess(label, 9)
        self.assertGreater(label, -1)
        self.assertEqual(type(label), int)
        self.assertEqual(bbox.shape, (4,))
        self.assertEqual(bbox.dtype, np.float64)

    def test_draw_random_t_shape_image(self):

        image = np.zeros([100, 100, 3], dtype=np.float32)
        image, boxes, polygon_masks, labels = random_rects.draw_random_t_shape_image(
            image=image,
            num_boxes = 10,
        )

    def test_create_dataset_generator(self):

        dataset = datasets_ops.from_numpy_generator(
            random_rects.create_random_rectangles_dataset_generator()
        )
        print(dataset)
