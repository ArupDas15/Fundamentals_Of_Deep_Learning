import tensorflow as tf
import numpy as np


def build_model(app, want_dense, k):
    tf.keras.backend.clear_session()
    if app.lower() == 'resnet50':
        base_model = tf.keras.applications.ResNet152V2(
            include_top=False,
            weights="imagenet",
            input_shape=(300, 300, 3),
            classes=1000,
            classifier_activation="softmax",
        )
    elif app.lower() == 'inceptionresnetv2':
        base_model = tf.keras.applications.InceptionResNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(300, 300, 3),
            classes=1000,
            classifier_activation="softmax",
        )
    elif app == 'inceptionV3':
        base_model = tf.keras.applications.InceptionV3(
            include_top=False,
            weights="imagenet",
            input_shape=(300, 300, 3),
            classes=1000,
            classifier_activation="softmax",
        )

    base_model.trainable = False
    total = len(base_model.layers)
    for i in range(total - k, total):
        base_model.layers[i].trainable = True
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.RandomCrop(height=300, width=300))

    model.add(tf.keras.Input(shape=(300, 300, 3)))
    model.add(base_model)
    model.add(tf.keras.layers.Flatten())
    if want_dense:
        model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model


import wandb
from wandb.keras import WandbCallback
from keras_preprocessing.image import ImageDataGenerator


def train():
    run = wandb.init()
    data_train = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        '/content/iNaturalist_Dataset/inaturalist_12K/train', shuffle=True,
        target_size=(400, 400))
    validation_data = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        '/content/iNaturalist_Dataset/inaturalist_12K/validate', shuffle=True,
        target_size=(400, 400))
    model = build_model(run.config.app, run.config.extra_dense, k=run.config.k)
    print(run.config.app)
    print(run.config.app == run.config.app)
    wandb.run.name = 'epoch_2_application_' + str(run.config.app) + '_dense_' + str(
        run.config.extra_dense) + '_k_' + str(run.config.k)

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=["accuracy"])
    model.fit(data_train, steps_per_epoch=len(data_train), epochs=2, validation_data=validation_data,
              callbacks=[WandbCallback()])
