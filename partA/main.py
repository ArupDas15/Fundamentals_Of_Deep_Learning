from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
import os
from keras.datasets import fashion_mnist
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, InputLayer
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import numpy as np


# placeholder - here I will get the data
# train = preprocess_img(datatype='train',batch_size=32,target_size=(126,126))
# validate=preprocess_img(datatype='validate',batch_size=32,target_size=(126,126))

# creating objet of sequential model


# this function builds the CNN with parameters passed by the users.
def build(input_shape, filters, size_filters, activation, optimiser, normalize, dropout_rate,neurons=1024):
    # iterating through filter and size_filters taking one from each at a time
    i = 0
    tf.keras.backend.clear_session()
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.RandomCrop(height=input_shape[0], width=input_shape[1],
                                                                    input_shape=(input_shape[0], input_shape[1], 3)))
    # model.add(tf.keras.Input(shape=input_shape))
    for (f, s) in zip(filters, size_filters):
        # Adding convulational layer with f filters and s as the poolsize
        model.add(Conv2D(f, s, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42)))
        if normalize:
            # adding batch normalization
            model.add(tf.keras.layers.BatchNormalization(momentum=.5))
        # adding relu layer after convulational layer
        model.add(Activation(activation))
        # adding max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    # flattenin the output of previous layers before passing to softmax
    model.add(Flatten())

    # adding dense layer with requisite number of neurons
    model.add(Dense(neurons, activation=activation))
    model.add(tf.keras.layers.Dropout(rate=dropout_rate))
    # output layer
    model.add(Dense(10, activation='softmax'))
    # compiling the whole thing
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=["accuracy"])
    return model


def train():
    run = wandb.init()

    start_filter = run.config.start_filter
    arch_parameter = run.config.arch_parameter
    filter_architecture = []
    for i in range(0, 5):
        filter_architecture.append(int(start_filter * arch_parameter ** i))
    print(filter_architecture)
    optimiser = tf.keras.optimizers.Adam(learning_rate=.0004)
    normalize = run.config.batch_normalize
    conv1 = run.config.conv1
    conv2 = run.config.conv2
    conv3 = run.config.conv3
    conv4 = run.config.conv4
    conv5 = run.config.conv5
    dropout_rate = 1 - run.config.dropout
    wandb.run.name = 'epoch_2_filter_size' + str(conv1) + '_' + str(conv2) + '_' + str(conv3) + '_' + str(
        conv4) + '_' + str(conv5) + '_drop_' + str(round(1 - dropout_rate, 4)) + '_normalize_' + str(
        run.config.batch_normalize) + '_start_fil_' + str(round(start_filter, 0)) + '_arch_param_' + str(
        round(arch_parameter, 4))

    model = build((300, 300, 3), filter_architecture,
                  [(conv1, conv1), (conv2, conv2), (conv3, conv3), (conv4, conv4), (conv5, conv5)], 32, 'relu',
                  optimiser=optimiser, normalize=normalize, dropout_rate=dropout_rate)
    # training
    dataset_augment = ImageDataGenerator(rescale=1. / 255)
    train = dataset_augment.flow_from_directory(os.path.join('/content/iNaturalist_Dataset/inaturalist_12K/train'),
                                                shuffle=True, target_size=(800, 800), batch_size=32)
    validate = ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        os.path.join('/content/iNaturalist_Dataset/inaturalist_12K/validate'), shuffle=False, target_size=(800, 800))
    model.fit(train, steps_per_epoch=len(train), epochs=2, validation_data=validate, callbacks=[WandbCallback()])
    model.save('/content/drive/MyDrive/Models/model.keras')
