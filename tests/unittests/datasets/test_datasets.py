import tensorflow as tf
import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.utils.testing_utils as utils
from keras_detection import ImageData


utils.maybe_enable_eager_mode()


def aug_fn(image_data: ImageData) -> ImageData:
    image = tf.cast(image_data.features.image, tf.float32)
    image = image / 255.0
    return image_data.replace_image(image)


class DatasetTest(tf.test.TestCase):
    def test_dataset_prepare_function(self):

        dataset = datasets_ops.from_numpy_generator(
            utils.create_fake_detection_dataset_generator(num_steps=20)
        )

        prepared_dataset = datasets_ops.prepare_dataset(
            dataset,
            model_image_size=(64, 48),
            augmentation_fn=aug_fn,
            num_epochs=2,
            batch_size=2,
            shuffle_buffer_size=1,
            prefetch_buffer_size=1,
        )

        for batch in prepared_dataset:
            batch = ImageData.from_dict(batch)
            self.assertEqual(batch.features.image.shape, [2, 64, 48, 3])
