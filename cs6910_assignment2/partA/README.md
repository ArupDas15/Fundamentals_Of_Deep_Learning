# Assignment 2: Part A: Training CNN from scratch
----------------------------------------------------
In this project we Build a small CNN model consisting of 5 convolution layers. Each convolution layer would be followed by a ReLU activation and a max pooling layer. 
# Libraries Used:
1. We have used keras and tensorflow to build the CNN model.
2. PIL library was used to handle images.
3. random library was used to select images from test set randomly.
4. os library was used to select images from the desired locations. 
5. matplotlib and mpl_toolkits library was used for plotting the (10,3) grid for the predicted images and  visualise all the filters in the first layer of our best model for a random image from the test set respectively.
6. Keras and tensorflow was used for getting the fashion mnist dataset.
7. numpy was used for some mathematical calculations and converting input arrays to types that were acceptible to keras and tensorflow functions. 
8. wandb was used to find the best hyperparameter configuratin and to make insightful observations from the plots obtained.
# Installations: #
1. We have used pip as the package manager. All the libraries we used above can be installed using the pip command.
2. Steps to Add Virtual Environment in IDE like Pycharm: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
# How to USE? #
The entire project has been modularised using functions and objects of classes to make it as scalable as possible for future developments and extensions.
To train a model the project makes a call to `train` in `main.py` file which inturn makes a call to `build()`. </br>
The parameters in build are the following <br />
arg1 : input_shape  : Size of the input image. For e.g. (300,300,3)<br />
arg2 : filters : This is a list which defines the number of filters per layer.<br />
arg3 : size_filters : This a list which contains the (width, height) of every filter in all the layer. The size of every filter in a layer is fixed.<br />
arg4 : activation : Accepts a string as input. Indicates the type of activation function used by the CNN which by default is set to`relu`<br />
arg6 : optimiser : The kind of optimiser algorithm to be used in CNN. It is of type optimizer ingerited from tensorflow.keras.optimizers<br />
arg7 : normalize : A boolean value (`True` or `False`) to denote if batch normalization needs to be performed for the input at every layer. If set to true the buold() performs batch normalization.<br />
arg8 : dropout_rate : The dropout rate signifies the probability of keeping a neuron. Hence the probability of dropping a neuron is 1-dropout_rate.<br />
arg9 : neurons : Denotes the number of neurons in the dense layer. Default value set to 1024.<br/>

The input at every 2d convolutional layer is randomly initialised using `xavier normal` initialisation with a seed value 42.<br />
## How is the model trained? ##
We utilised the RandomCrop class in Keras which randomly crops all images belonging to the same batch to the same cropping location. This proved to be a good substitute for Data augmentation as it helped to improve the robustness of the model.
We take the images from the train directory and set it to 400 * 400 size and fit the training images on the compiled model using `model.fit()`.
</br>***Example***:</br> 
```model.fit(train, steps_per_epoch=len(train), epochs=2, validation_data=validate, callbacks=[WandbCallback()])```</br>

## How to change  number of filters, size of filters and activation function? ##
To change  the number of filters, size of filters and activation function we can pass the required parameters to `build()` in train() of main.py.
</br>***Example***:</br>
```build((300, 300, 3), [128, 160, 200, 250, 312],[(5, 5), (7, 7), (7, 7), (3, 3), (5, 5)], 'relu',optimiser=tensorflow.keras.optimizers.Adam(learning_rate=.0004), normalize=True, dropout_rate=1,neurons=1024)```
## How is the model evaluated? ##
The images from the test directory are chosen randomly and stretched to (400,400) dimension irrespective of the original size of the image. The best model is loaded and the accuracy on the model is evaluated by calling `model.evaluate()` to report the accuracy.
</br>***Example***</br>
To load the best model.</br>
```best_model = keras.models.load_model('/content/best_model.keras')```</br>
To report the accuracy.</br>
```score, acc = best_model.evaluate(test, batch_size=32, verbose=0)```</br>
```print('Test accuracy:', acc)```</br>

# Acknowledgements #
1. The entire project has been developed from the lecture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. CNN using tensor flow: https://www.tensorflow.org/tutorials/images/cnn
3. CNN 2D class description: https://www.geeksforgeeks.org/keras-conv2d-class/
4. https://wandb.ai
5. Why do CNNs require a fixed input size?: https://arxiv.org/pdf/1406.4729.pdf
6. Paper on Dropout : http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
7. Guided backpropagation: https://arxiv.org/pdf/1412.6806.pdf

