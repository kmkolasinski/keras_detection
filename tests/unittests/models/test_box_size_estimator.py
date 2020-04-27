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


def aug_fn(image_data: ImageData) -> ImageData:
    image = tf.cast(image_data.features.image, tf.float32)
    image = tf.image.random_brightness(image, max_delta=1.2)
    return image_data.replace_image(image)


class SizeEstimatorBackboneTest(tf.test.TestCase):

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
        model.fit(prepared_train_dataset, epochs=1, steps_per_epoch=5)
        model.save_weights(f"{self.test_dir}/model.h5")

        builder.convert_to_tflite(
            f"{self.test_dir}/model.h5",
            save_path=f"{self.test_dir}/model.tflite",
            export_batch_size=1,
            raw_dataset=raw_dataset,
            num_dataset_samples=2,
            num_test_steps=1,
            merge_feature_maps=True,
            postprocess_outputs=True,
        )

        qmodel = builder.build_quantized(
            non_quantized_model_weights=f"{self.test_dir}/model.h5"
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, beta_2=0.99)
        qmodel.compile(optimizer, **builder.get_model_compile_args())
        qmodel.fit(prepared_train_dataset, epochs=1, steps_per_epoch=5)

        builder.convert_to_tflite(
            qmodel,
            save_path=f"{self.test_dir}/qmodel.tflite",
            export_batch_size=1,
            raw_dataset=raw_dataset,
            num_dataset_samples=2,
            num_test_steps=1,
            merge_feature_maps=True,
            postprocess_outputs=True,
        )

    def test_create_box_size_estimator_backbone(self):
        image_dim = 128
        backbone = resnet.ResNetBackbone(
            input_shape=(None, None, 3),
            units_per_block=(1, 1),
            num_last_blocks=1,
        )

        se_backbone = bse.SizeEstimatorBackbone(
            backbone.backbone,
            (image_dim, image_dim, 3), num_scales=3
        )

        se_backbone_model = se_backbone.as_model()
        tflite_ops.TFLiteModel.from_keras_model(se_backbone_model)
        se_backbone_model.summary()

        se_backbone_model = se_backbone.as_model(quantized=True)
        tflite_ops.TFLiteModel.from_keras_model(se_backbone_model)

    def test_train_and_export_model(self):

        image_dim = 64
        backbone = resnet.ResNetBackbone(
            input_shape=(None, None, 3),
            units_per_block=(1, 1),
            num_last_blocks=1,
        )
        num_scales = 3

        se_builder = bse.BoxSizeEstimatorBuilder(
            (image_dim, image_dim, 3),
            backbone.backbone,
            box_size_task= bse.get_mean_box_size_task(),
            objectness_task=bse.get_objectness_task(),
            num_scales=num_scales
        )

        self.train_export_model(se_builder)
