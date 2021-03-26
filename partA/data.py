from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import glob
import numpy as np

def resize(datatype,target_size=(240,240)):
    desired_size = target_size[0]
    base_path = os.path.join(os.path.dirname(os.getcwd()),'iNaturalist_Dataset','inaturalist_12K',datatype)
    # Get the list of folders in the directory datatype
    class_labels = os.listdir(base_path)
    #print(class_labels)
    label_dict = {class_labels[i]: class_labels.index(class_labels[i]) for i in range(0, len(class_labels))}
    #print(label_dict)
    data = []
    data_label = []
    for key,value in label_dict.items():
        class_folder = base_path+os.sep+key+os.sep
        if datatype == 'train':
            os.makedirs(os.path.join(os.path.dirname(os.getcwd()),'Output',datatype,key))
            i = 1
        for file in glob.glob(class_folder + "*"):
            im = Image.open(file)
            current_size = im.size
            ratio = float(desired_size)/max(current_size)
            # resize the input image so that its maximum size equals to the given size.
            new_size = (int(ratio * current_size[0]),int(ratio * current_size[1]))
            # Resize the current image
            im = im.resize(new_size, Image.ANTIALIAS)
            # create a new image of desired size and paste the resized image on the black image
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size-new_size[0])//2,
                            (desired_size-new_size[1])//2))
            if datatype == 'train':
                file_name = os.path.join(os.path.dirname(os.getcwd()),'Output',datatype,key,str(i)+'.jpg')
                new_im.save(file_name)
                i = i + 1
            img = np.array(new_im)
            data.append(img)
            data_label.append(value)
    if datatype != 'train':
        return np.array(data),np.array(data_label)

def preprocess_img(datatype,batch_size=64,target_size=(240,240)):
    # Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory
    dataset = ImageDataGenerator(rescale=1./255)
    # load and iterate training dataset. To evaluate the model set ‘shuffle‘ to ‘False.’
    if datatype != 'train':
        x,y =resize(datatype,target_size=target_size)
        data = dataset.flow(x,y,batch_size=batch_size,shuffle=False)
    else:
        resize(datatype, target_size=target_size)
        data = dataset.flow_from_directory(os.path.join(os.path.dirname(os.getcwd()),'Output',datatype), class_mode='categorical', batch_size=batch_size, target_size=target_size,shuffle=False)
    return data

# train = preprocess_img(datatype='test',batch_size=32,target_size=(240,240))
# print(train.shape)
#validate=preprocess_img(datatype='validate',batch_size=32,target_size=(240,240))
# type of validate object returned
# <class 'keras_preprocessing.image.numpy_array_iterator.NumpyArrayIterator'>
