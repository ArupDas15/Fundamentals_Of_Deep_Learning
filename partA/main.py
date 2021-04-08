from keras.models import Sequential
from keras_preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.layers import Dense, Flatten, InputLayer
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback



path='/content/iNaturalist_Dataset/inaturalist_12K/'


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

def train():
    run = wandb.init()
    opti = None
    #wandb.run.name = run.config.optimiser+str(np.random.rand(1))
    if run.config.filter_architecture == 'increasing':
      filter_architecture=[64,128,256,350,380]
    elif run.config.filter_architecture == 'decreasing':
      filter_architecture=[64,70,80,90,256]
    elif run.config.filter_architecture == 'equal':
      filter_architecture= [64,70,80,90,128]
    print(filter_architecture)
    optimiser = tf.keras.optimizers.Nadam(learning_rate=.0007)
    normalize=False
    if run.config.batch_normalize == 'Yes':
        normalize=True
    dropout =False
    dropout_rate=0
    if run.config.dropout != 'None':
      dropout_rate=run.config.dropout
      dropout=True
    model=build((299,299,3),filter_architecture,[(3,3),(3,3),(3,3),(3,3),(3,3)],32,'relu',optimiser=optimiser,normalize=normalize,dropout=dropout,dropout_rate=dropout_rate)
    #training
    dataset_augment = ImageDataGenerator(rescale=1. / 255,rotation_range=20, width_shift_range=0.1, height_shift_range=0.1,
                         shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                         fill_mode="nearest")
    train=dataset_augment.flow_from_directory(os.path.join(path,'train'), shuffle=True, target_size=(600,600),batch_size=32)
    validate=ImageDataGenerator(rescale=1. / 255).flow_from_directory(os.path.join(path,'validate'), shuffle=False, target_size=(600,600))
    model.fit(train ,steps_per_epoch=len(train),epochs=10, validation_data=validate,callbacks=[WandbCallback()])
    model.save('/content/drive/MyDrive/Models/model.keras')