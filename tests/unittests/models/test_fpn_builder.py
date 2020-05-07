import shutil
import tempfile

import keras_detection.datasets.datasets_ops as datasets_ops
import keras_detection.utils.testing_utils as utils
import tensorflow as tf
from keras_detection import FPNBuilder
from keras_detection import ImageData
from keras_detection.backbones import resnet
from keras_detection.backbones.fpn import FPNBackbone
from keras_detection.backbones.simple_cnn import SimpleCNNBackbone
from keras_detection.tasks import standard_tasks
from keras_detection.utils import tflite_debugger
import numpy as np


def aug_fn(image_data: ImageData) -> ImageData:
    image = tf.cast(image_data.features.image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=1.2)
    return image_data.replace_image(image)


class BuilderTest(tf.test.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def train_export_model(self, builder: FPNBuilder):

        image_dim = builder.input_shape[1]
        raw_dataset = datasets_ops.from_numpy_generator(
            utils.create_fake_detection_dataset_generator(num_steps=100)
        )

        train_dataset = datasets_ops.prepare_dataset(
            raw_dataset,
            model_image_size=(image_dim, image_dim),
            augmentation_fn=aug_fn,
            num_epochs=-1,
            batch_size=2,
            shuffle_buffer_size=1,
            prefetch_buffer_size=1,
        )
        model = builder.build()

        prepared_train_dataset = train_dataset.map(
            builder.get_build_training_targets_fn()
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_2=0.99)
        model.compile(optimizer, **builder.get_model_compile_args())
        model.fit(prepared_train_dataset, epochs=1, steps_per_epoch=2)
        model.save_weights(f"{self.test_dir}/model.h5")

        builder.convert_to_tflite(
            f"{self.test_dir}/model.h5",
            save_path=f"{self.test_dir}/model.tflite",
            export_batch_size=1,
            raw_dataset=raw_dataset,
            num_dataset_samples=1,
            num_test_steps=1,
            merge_feature_maps=True,
            postprocess_outputs=True,
            convert_quantized_model=True
        )

        qmodel = builder.build_quantized(
            non_quantized_model_weights=f"{self.test_dir}/model.h5"
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_2=0.99)
        qmodel.compile(optimizer, **builder.get_model_compile_args())
        qmodel.fit(prepared_train_dataset, epochs=1, steps_per_epoch=2)

        builder.convert_to_tflite(
            qmodel,
            save_path=f"{self.test_dir}/qmodel.tflite",
            export_batch_size=1,
            raw_dataset=raw_dataset,
            num_dataset_samples=1,
            num_test_steps=1,
            merge_feature_maps=True,
            postprocess_outputs=True,
        )

    def test_model_with_objectness(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )
        tasks = [standard_tasks.get_objectness_task()]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        self.train_export_model(builder)

    def test_model_with_box_shape(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=1,
        )
        tasks = [standard_tasks.get_box_shape_task()]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        self.train_export_model(builder)

    def test_model_with_objectness_single_feature_map(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=1,
        )
        tasks = [standard_tasks.get_objectness_task()]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        self.train_export_model(builder)

    def test_model_with_many_tasks_many_fms(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1, 1),
            num_last_blocks=2,
        )
        tasks = [
            standard_tasks.get_objectness_task(),
            standard_tasks.get_box_shape_task(),
            standard_tasks.get_multiclass_task(num_classes=10),
        ]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        self.train_export_model(builder)

    def test_simple_cnn_model_with_many_tasks_many_fms(self):

        image_dim = 64
        backbone = SimpleCNNBackbone(
            input_shape=(image_dim, image_dim, 3), init_filters=16, num_last_blocks=1
        )
        tasks = [
            standard_tasks.get_objectness_task(),
            standard_tasks.get_box_shape_task(),
            standard_tasks.get_multiclass_task(
                num_classes=10, fl_gamma=0, activation="softmax"
            ),
        ]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        self.train_export_model(builder)

    def test_model_with_objectness_and_fpn_backbone(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1, 1),
            num_last_blocks=3,
        )
        backbone = FPNBackbone(
            backbone, depth=16,
            num_first_blocks=1
        )

        tasks = [standard_tasks.get_objectness_task()]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        self.train_export_model(builder)


class TFLiteDebuggerTest(tf.test.TestCase):
    def test_debug_quantized_model(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(image_dim, image_dim, 3),
            units_per_block=(1, 1),
            num_last_blocks=2,
        )
        tasks = [standard_tasks.get_objectness_task()]

        builder = FPNBuilder(backbone=backbone, tasks=tasks)
        builder.build()

        image_dim = builder.input_shape[1]
        raw_dataset = datasets_ops.from_numpy_generator(
            utils.create_fake_detection_dataset_generator(num_steps=100)
        )

        train_dataset = datasets_ops.prepare_dataset(
            raw_dataset,
            model_image_size=(image_dim, image_dim),
            augmentation_fn=aug_fn,
            num_epochs=-1,
            batch_size=2,
            shuffle_buffer_size=1,
            prefetch_buffer_size=1,
        )
        prepared_train_dataset = train_dataset.map(
            builder.get_build_training_targets_fn()
        )

        def representative_dataset():
            for features, labels in prepared_train_dataset:
                for image in features["image"]:
                    yield np.expand_dims(image, 0)


        quantized_model = builder.build_quantized()

        outputs_diffs = tflite_debugger.debug_model_quantization(
            representative_dataset(), quantized_model, max_samples=1
        )
        print(outputs_diffs)
