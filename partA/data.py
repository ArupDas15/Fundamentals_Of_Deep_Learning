from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import os
def preprocess_img(datatype,batch_size=64,target_size=(240,240)):
    # Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory
    dataset = ImageDataGenerator(rescale=1./255)
    basepath = os.path.join('..','iNaturalist_Dataset','inaturalist_12K')
    if datatype=='train':
        # load and iterate training dataset. To evaluate the model set ‘shuffle‘ to ‘False.’
        train_it = dataset.flow_from_directory(os.path.join(basepath,'train'), class_mode='categorical', batch_size=batch_size, target_size=target_size,shuffle=False)
        return train_it
    elif datatype=='validate':
        # load and iterate validation dataset
        val_it = dataset.flow_from_directory(os.path.join(basepath,'validate'), class_mode='categorical', batch_size=64, target_size=(240,240))
        return val_it
    else:
        # load and iterate test dataset
        test_it = dataset.flow_from_directory(os.path.join(basepath,'test'), class_mode='categorical',
                                              batch_size=64, target_size=(240, 240))
        return test_it

    return



