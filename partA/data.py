from keras_preprocessing.image import ImageDataGenerator
from PIL import Image
import os
import glob
import numpy as np
from os import path


# Reference: https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
def resize(datatype, target_size=(240, 240)):
    desired_size = target_size[0]
    base_path = os.path.join(os.path.dirname(os.getcwd()), 'iNaturalist_Dataset', 'inaturalist_12K', datatype)
    # Get the list of folders in the directory datatype
    class_labels = os.listdir(base_path)
    # Assigns a numerical value to every distinct class in the dataset by creating a dictionary.
    label_dict = {class_labels[i]: class_labels.index(class_labels[i]) for i in range(0, len(class_labels))}
    # data is a list which stores the numpy array of every image
    data = []
    # data_label is a list which stores the categorical label corresponding to every image matrix in data
    data_label = []
    for key, value in label_dict.items():
        class_folder = base_path + os.sep + key + os.sep
        if datatype == 'train':
            os.makedirs(os.path.join(os.path.dirname(os.getcwd()), 'Output', datatype, key))
            i = 1
        for file in glob.glob(class_folder + "*"):
            im = Image.open(file)
            current_size = im.size
            # We divide by the maximum dimension so that the resized image does not extend beyond the desired size
            ratio = float(desired_size) / max(current_size)
            """ Resize the input image so that its maximum side is equal to the target dimension.
                If maximum dimension equals the given dimension then all other dimension will also fit in the square.
            """
            new_size = (int(ratio * current_size[0]), int(ratio * current_size[1]))
            # Resize the current image
            im = im.resize(new_size, Image.ANTIALIAS)
            # create a new image of desired size
            new_im = Image.new("RGB", (desired_size, desired_size))
            """ paste the resized image on the black image obtained above. We are dividing the margin by 2 and taking it
            floor. We are doing this there should be equal amount of margin on both sides."""
            new_im.paste(im, ((desired_size - new_size[0]) // 2,
                              (desired_size - new_size[1]) // 2))
            if datatype == 'train':
                file_name = os.path.join(os.path.dirname(os.getcwd()), 'Output', datatype, key, str(i) + '.jpg')
                new_im.save(file_name)
                i = i + 1
            img = np.array(new_im)
            data.append(img)
            data_label.append(value)
    if datatype != 'train':
        return np.array(data), np.array(data_label)


# Reference: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory
def preprocess_img(datatype, batch_size=64, target_size=(240, 240)):
    if datatype != 'train':
        re_scaler = ImageDataGenerator(rescale=1. / 255)
        # x is a numpy array of image data and y is a numpy array of corresponding labels.
        x, y = resize(datatype, target_size=target_size)
        # load and iterate over validation or test data.
        data = re_scaler.flow(x, y, batch_size=batch_size)
    else:
        """ Since the training data is large in size so we save the processed images in Output folder before doing any further computations.
            This workaround is done because on the fly computation could lead to memory error due to large training data. 
        """
        dataset_augmentar = ImageDataGenerator(rescale=1. / 255, rotation_range=30,
                                               width_shift_range=0.3,
                                               height_shift_range=0.2,
                                               shear_range=0.1,
                                               zoom_range=0.3)
        """ if Output data is available don't resize again. If it's not
         available use training data to get it's resized version."""
        if not path.exists(os.path.join(os.path.dirname(os.getcwd()), 'Output', datatype)):
            resize(datatype, target_size=target_size)
        # load and iterate over training dataset. To evaluate the model set ‘shuffle‘ to ‘False.’
        data = dataset_augmentar.flow_from_directory(os.path.join(os.path.dirname(os.getcwd()), 'Output', datatype),
                                                     class_mode='categorical', batch_size=batch_size,
                                                     shuffle=True)
    return data
