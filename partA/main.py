from keras.models import Sequential
# import os
# print(os.listdir("../iNaturalist_Dataset/inaturalist_12K/train/"))
from keras.datasets import fashion_mnist
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, InputLayer
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
import tensorflow as tf
import numpy as np
from data import *

#placeholder - here I will get the data
train = preprocess_img(datatype='train',batch_size=32,target_size=(240,240))
validate=preprocess_img(datatype='validate',batch_size=32,target_size=(240,240))

#creating objet of sequential model
model = Sequential()

#this function builds the CNN with parameters passed by the users.
def build(filters,size_filters,neurons,activation):
  # iterating through filter and size_filters taking one from each at a time
  for (f,s) in zip(filters,size_filters):
    #Adding convulational layer with f filters and s as the poolsize
    model.add(Conv2D(f, s))
    #adding relu layer after convulational layer
    model.add(Activation(activation))
    #adding max pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
  #adding dense layer with requisite number of neurons
  model.add(Dense(neurons,activation=activation))
  #flattenin the output of previous layers before passing to softmax
  model.add(Flatten())
  #output layer
  model.add(Dense(10,activation='softmax'))
  #compiling the whole thing
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',metrics=["accuracy"])


build([16,16,16,16,16],[(3,3),(3,3),(3,3),(3,3),(3,3)],10,'relu')
#training
model.fit( train,
        steps_per_epoch=len(train),
        epochs=1)
#the model summary
model.summary()