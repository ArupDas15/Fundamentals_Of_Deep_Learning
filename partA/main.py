#used to initialize NN
from keras.models import Sequential
#used for convolution step which makes convolution layer
from keras.layers.convolutional import Conv2D
# used to add layers (hidden layer and outputs)
from keras.layers import Dense
# used to convert pooled features into vector that will be input to NN
from keras.layers import Flatten
# used for pooling step which makes pooling layer
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
import tensorflow as tf
from partA.data import *

#placeholder - here I will get the data
train = preprocess_img(datatype='train',batch_size=32,target_size=(240,240))
validate=preprocess_img(datatype='validate',batch_size=32,target_size=(240,240))

#creating an object of sequential model
model = Sequential()

#this function builds the CNN with parameters passed by the users.
def build(filters,size_filters,neurons,activation,input_shape):
  # Flag to check if it is the first convolution layer
  first_conv_layer = 0
  # iterating through filter and size_filters taking one from each at a time
  for (f,s) in zip(filters,size_filters):
    # Adding convulational layer with f filters and s as the poolsize
    if first_conv_layer == 0:
      model.add(Conv2D(filters=f,kernel_size=s,input_shape=input_shape))
      first_conv_layer = first_conv_layer + 1
    else:
      model.add(Conv2D(filters=f,kernel_size=s))
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
  model.compile(optimizer=tf.keras.optimizers.Adam(), loss='categorical_crossentropy',metrics=["accuracy"])


build([16,16,16,16,16],[(3,3),(3,3),(3,3),(3,3),(3,3)],10,'relu',input_shape=(240,240,3))
#training
model.fit( train,
        steps_per_epoch=len(train),
        epochs=1)
#the model summary
model.summary()
