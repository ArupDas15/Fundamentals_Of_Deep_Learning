# Assignment 2: Part B: Fine-tuning a pre-trained model
--------------------------------------------------------
In this project instead of training a model from scratch, we use a model pre-trained on a similar/related task/dataset. We load any model from say InceptionV3, InceptionResNetV2, ResNet50, Xception, etc which are pre-trained on the ImageNet dataset and then fine-tune it using the naturalist data.
# Libraries Used: 
1. We have used tensorflow and keras to use the pre-trained model.
2. numpy was used for some mathematical calculations and converting input arrays to types that were acceptible to keras and tensorflow functions. 
3. The wandb callback feature supporting keras was used to log the metrics and loss values tracked in model.fit.
# Installations: #
1. We have used pip as the package manager. All the libraries we used above can be installed using the pip command
2. Steps to Add Virtual Environment in IDE like Pycharm: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
# How to USE? #
The entire project has been modularised using functions and objects of classes to make it as scalable as possible for future developments and extensions.
The python file `transfer.py` contains the `train()` which first chooses the pre-trained model to load by making a call to `build_model()`. 
The parameters in `build.model()` are the following: <br />
arg1 : app  : Accepts a string with possible values as 'resnet50','inceptionresnetv2','inceptionV3' in any case. <br />
arg2 : want_dense : Accepts a boolean value (True/False).<br />
arg3 : k : Denotes that the last k layers in the network are freezed.

The data is then fit into the model using `model.fit()`. </br>

***Example***:</br> 
```model.fit(data_train, steps_per_epoch=len(data_train), epochs=2, validation_data=validation_data,callbacks=[WandbCallback()])```</br>
where ```data_train = ImageDataGenerator(rescale=1. / 255).flow_from_directory('/content/iNaturalist_Dataset/inaturalist_12K/train', shuffle=True,target_size=(400,400))```
</br>Here the images in data_train are loaded from the train directory by rescaling every taining image by a factor of 1/255. This brongs the RGB coefficients of the image in a scale between 0 to 1 making it easier to perform mathematical calculations. 
# Acknowledgements #
1. The entire project has been developed from the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. ImageNet Classification with Deep Convolutional Neural Networks https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
3. What is pre training? https://martin-thoma.com/ml-glossary/#pre-training
4. https://wandb.ai
5. https://stackoverflow.com/questions/53503389/how-to-set-parameters-in-keras-to-be-non-trainable


