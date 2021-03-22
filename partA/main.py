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
model = Sequential()
def build(filters,size_filters,neurons,activation):
  for (f,s) in zip(filters,size_filters):
    model.add(Conv2D(f, s))
    model.add(Activation(activation))
    model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dense(neurons,activation=activation))
  model.add(Flatten())
  model.add(Dense(10,activation='softmax'))
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])


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

#placeholder - here I will get the data
x_train=np.random.rand(1000,240,240,3)
y_train=np.random.rand(1000)
x_validate=np.random.rand(100,240,240,3)
y_validate=np.random.rand(100)
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
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=["accuracy"])


build([16,16,16,16,16],[(3,3),(3,3),(3,3),(3,3),(3,3)],10,'relu')
#training
model.fit(x=x_train,y=y_train,epochs=1)
#the model summary
model.summary()