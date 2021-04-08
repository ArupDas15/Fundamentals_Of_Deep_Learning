from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
# import os
# print(os.listdir("../iNaturalist_Dataset/inaturalist_12K/train/"))
from keras.datasets import fashion_mnist
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, InputLayer
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
import tensorflow as tf
import numpy as np





# this function builds the CNN with parameters passed by the users.
def build(input_shape, filters, size_filters, neurons, activation, optimiser, normalize, dropout, dropout_rate):
    # iterating through filter and size_filters taking one from each at a time
    i = 0
    tf.keras.backend.clear_session()
    model = Sequential()
    model.add(tf.keras.layers.experimental.preprocessing.RandomCrop(height=input_shape[0], width=input_shape[1]))
    model.add(tf.keras.Input(shape=input_shape))
    for (f, s) in zip(filters, size_filters):
        # Adding convulational layer with f filters and s as the poolsize
        model.add(Conv2D(f, s, kernel_initializer=tf.keras.initializers.GlorotNormal(seed=42)))
        if normalize:
            # adding batch normalization
            model.add(tf.keras.layers.BatchNormalization())
        # adding relu layer after convulational layer
        model.add(Activation(activation))
        # adding max pooling layer
        model.add(MaxPooling2D(pool_size=(2, 2)))
        if dropout:
            # adding dropout to first and second CNN block only.
            if i == 0 or i == 1:
                model.add(tf.keras.layers.Dropout(rate=1 - dropout_rate))

        i = i + 1
    # flattening the output of previous layers before passing to softmax
    model.add(Flatten())

    # adding dense layer with requisite number of neurons
    model.add(Dense(512, activation=activation))
    # output layer
    model.add(Dense(10, activation='softmax'))
    # compiling the whole thing
    model.compile(optimizer=optimiser, loss='categorical_crossentropy', metrics=["accuracy"])
    return model
