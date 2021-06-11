# Assignment 1: Neural Network model development
----------------------------------------------------
In this project we implement a feed forward neural network and use gradient descent (and its variants: momentum, NAG, RMSProp, ADAM, NADAM) with backpropagation for classifying images from the fashion-mnist data over 10 class labels. We use wandb for visualisation of data for training, model comparison and accuracy of a large number of experiments that we have performed to make meaningful inferences.
# Libraries Used:
1. We have used numpy for all the mathematical calculations in forward propagation, back propagation algorithm and loss (cross entropy and squared error) function computation.
2. Scikit learn library was used to generate the confusion matrix which was converted into a dataframe using pandas library. 
3. Seaborn and matplotlib libraries were used for plotting the confusion matrix.
4. Keras and tensorflow was used for getting the fashion mnist dataset.
5. Pickle was used to save the best neural network model obtained during training.
# Installations: #
1. We have used pip as the package manager. All the libraries we used above can be installed using the command: `pip install -r requirements.txt`
2. Steps to Add Virtual Environment in IDE like Pycharm: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env
# How to USE? #
The entire project has been modularised using functions and classes to make it as scalable as possible for future developments and extensions.
To train a model the project makes a call to `master()` in `main.py` file. </br>
The parameters in master are the following <br />
arg1 : batch : Number of datapoints to be in one batch. For e.g. 16, 32, 64<br />
arg2 : epochs : Number of passes to be done over the entire data<br />
arg3 : output_dim : Number of classes in the classification dataset<br />
arg4 : activation : The type of activation function used. The possible values are anyone one from `sigmoid, tanh, relu` <br />
arg5 : opt : An object of variants of gradient descents. The objects can be of type `SimpleGradientDescent`, `MomentumGradientDescent`, `NAG`, `RMSProp`, `ADAM`, `NADAM`<br />
arg6 : layer_1 : Number of neurons in layer 1.<br />
arg7 : layer_2 : Number of neurons in layer 2.<br />
arg8 : layer_3 : Number of neurons in layer 3.<br />
arg9 : loss : The loss function to be used for the classification dataset. The loss functions currently supported are `squared_error` and `cross_entropy`. <br />
arg10 : weight_init : Possible values are `random`, `xavier` for random weight initialisation and Xavier weight initialisation respectively.<br />
## Object initialization of optimiser classes ##
As mentioned above opt is of type `SimpleGradientDescent`, `MomentumGradientDescent`, `NAG`, `RMSProp`, `ADAM`, `NADAM`. Here we look at how opt can be initialized with respect to different optimiser classes supported.</br>
1. opt = SimpleGradientDescent(eta= <enter learning rate value>, layers = <enter number of layers in the network>, weight_decay= <enter the weight decay value: Default value 0>)
2. opt = MomentumGradientDescent(eta= <enter learning rate value>, layers = <enter number of layers in the network>, gamma = <enter the gamma value>, weight_decay= <enter the weight decay value: Default value 0>)
3. opt = NAG(eta= <enter learning rate value>, layers = <enter number of layers in the network>, gamma = <enter the gamma value>, weight_decay= <enter the weight decay value: Default value 0>)
4. opt = RMSProp(eta= <enter learning rate value>, layers = <enter number of layers in the network>, beta = <enter the beta value>, weight_decay= <enter the weight decay value: Default value 0>)
5. opt = ADAM(eta= <enter learning rate value>, layers = <enter number of layers in the network>, weight_decay= <enter the weight decay value: Default value 0>, beta1 = <enter the beta value: Default value 0.9>, beta2 = <enter the beta value: Default value 0.999, eps = <enter the epsilon value: Default value 1e-8>>) 
6. opt = NADAM(eta= <enter learning rate value>, layers = <enter number of layers in the network>, weight_decay= <enter the weight decay value: Default value 0>, beta1 = <enter the beta value: Default value 0.9>, beta2 = <enter the beta value: Default value 0.999, eps = <enter the epsilon value: Default value 1e-8>)</br> 
***Example***:</br> 
```master(batch=495, epochs=7, output_dim=10, activation='tanh', opt=ADAM(eta=0.003576466933615937,layers=4,weight_decay=0.31834976996809683), layer_1 = 32,layer_2 = 64 ,layer_3 = 16 , loss='squared_error',weight_init='xavier')```</br>
</br> Here we have initialized an opt object of type ADAM having learning rate as 0.003576466933615937 and weight decay set to 0.31834976996809683. We are going to train a neural network of 4 layers where layer 1 consists of 32 neurons, layer 2 consists of 64 neurons and layer 3 consists of 16 neurons. The loss function used is a squared entropy loss function and the weights have been initialized using Xavier initialisation.
## How to change the number of layers, the number of neurons in each layer and the type of activation function used in each layer? ##
The *add_layer()* in master() provides the flexibility to change the number of layers, the number of neurons in each layer and the type of activation function used in each layer.</br></br>
```add_layer(number_of_neurons, context, weight_init, input_dim=None)```
</br>***Parameters*** : </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**number_of_neurons** : number of neurons to be added in the layer.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**context** : Accepts parameters of type string. Applies the type of activation function needed.  Supported values: sigmoid, tanh, relu </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**weight_init** : Accepts parameters of type string. Performs either random or Xavier weight initialisation. Supported values: random, xavier </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**input_dim** : The input dimension of the datset. Required only for the input layer. If not given, then default value is set to None. </br>

***Example***: 
</br>```add_layer(number_of_neurons=layer_1, context=activation, input_dim=784, weight_init=weight_init)``` </br>
where ```layer_1 = 32, activation = 'tanh', weight_init='xavier'```
</br> Hence in order to change the configuration of the neural network we only need to chenge the arguments passed to two functions:
1. master()
2. add_layer() inside master()
# How to add a new optimisation algorithm to the existing code? #
The current code structure is fexible enough to accodomodate new optimisation algorithms. To add a new optimsation algorithm such as Eve one has to add a new class in the python file `optimiser.py`. The new class should primarily contain the `__init__()` for initializing various hyperparameter values such as the learning rate(eta), learning rate controller (lrc) if needed in order to perform learning rate annealing, weight decay to obtain regularised loss function and the `descent()` which would accept three parameters `self,network,gradient` where self is an object of type ofoptimsation algorithm used, network is a list which contains the entire neural network and gradient is a list which contains the current gradient with respect to the parameters in the output layer (y<sup>hat</sup> and a<sub>i</sub>), previous hidden layers (h<sub>i</sub> and a<sub>i</sub>), weight and biases.</br></br>
gradient[i]['weight']: Denotes the weight matrix `W` at layer `i`.</br>
gradient[i]['bias']: Denotes the `bias` at layer `i`.</br>
gradient[i]['h']: Denotes the gradient with respect to the post activation output `h`<sub>i</sub> at layer `i`.</br>
gradient[i]['a']: Denotes the gradient with respect to the pre activation output `a`<sub>i</sub> at layer `i`.</br>
The descent algorithm would typically implement the update to be done on the weight matrix and bias at a layer i.</br>
Now the newly implemented optimzation algorithm's instance can be passed to assiged to opt when making call to master. This has been described [above](https://github.com/utsavdey/cs6910_assignment1/blob/master/README.md#how-to-use). 
## Forward Propagation Procedure ##
*Desciption*: Performs forward propagation on the data point.</br>
```forward_propagation(n, x)```</br>
***Parameters***</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**n** : number of layers in the network. This includes the input layer, number of hidden layers in the neural network and the output layer.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**x**: Accepts a single data vector having normalised color values between 0 to 1.</br>
## Backward Propagation Procedure ##
*Desciption*: Performs backward propagation on the data point.</br>
Parameters : </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**number_of_layers** : number of layers in the network. This includes the input layer, number of hidden layers in the neural network and the output layer.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**x** : Accepts a single data vector having normalised color values between 0 to 1.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**y** : Integer value. Denotes class labels in the rage 0-9.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**number_of_datapoint** : batch size of the training data.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**loss_type** :  Accepts parameters of type string. Acceptable values are squared_error and cross_entropy.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**clean** : A boolean flag used to indicate whether it is the first datapoint from the batch by setting it to True or not. If not specified, then default value is set to False.</br>
# Construction of Neural Network #
The fit() defined in main.py and called by master() is responsible for training the neural network. It internally makes a call for forward propagation, backward propagation for a datapoint and computes validatation and testing accuracy and loss after every epoch and logs the result to wandb.</br>
```fit(datapoints, batch, epochs, labels, opt, loss_type, augment)```</br></br>
*Description*:
1. Separates training and validation set.
2. Performs dataset augmentation. 
3. Randomly samples data points from train data to form batches of batch size = batch. 
4. Performs forward propagation, backward propagation for each of the data points in the batch. 
5. Calculates the validation loss, validation accuracy , training loss and training accuracy after every epoch. An epoch is defined as one pass over the entire data. </br>
Parameters : </br> 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**datapoints** :  uint8 arrays of grayscale image data with shape (num_samples, 28, 28). </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**batch** : Number of datapoints to be in one batch. For e.g. 16, 32, 64. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**epochs** : Number of passes to be done over the entire data. </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**labels** : uint8 arrays of labels (integers in range 0-9) with shape (num_samples).</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**opt** : An object of variants of gradient descents. The objects can be of type `SimpleGradientDescent`, `MomentumGradientDescent`, `NAG`, `RMSProp`, `ADAM`, `NADAM` </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**loss_type** : Accepts parameters of type string. Acceptable values are squared_error and cross_entropy.</br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**augment** : Adds data points to training data for dataset augmentation. This is of type int.</br>
# Program Flow #
The program for training the neural network begins by making call to *train()*. The train() passes parmeters to *master()* depending on the configurations provided to configuration.yaml file in wandb for creating a sweep. The master() is responsible for adding layers to the neural network by making calls to *add_layer()* and initializing the training process by calling the *fit()* where forward and backward propagations for every training data point are made with respect to the optimzation algorithm used. The optimization algorithm to be used is passed to opt as an object as described [above](https://github.com/utsavdey/cs6910_assignment1/blob/master/README.md#how-to-use). After every epoch the validation loss, validation accuracy, training loss and training accuracy is logged in wandb for effective data visualization. The program terminates with master returning the trained neural network. To save the trained neural network model run the code in save_model.py file by providing appropriate values to master(). </br></br>
P.S. To skip writing observations into wandb comment the train() and wandb.log() statements in main.py file and pass arguments directly to master(). The same procedure to be followed to save a neural network model using the save_model.py file.
# Acknowledgements #
1. The entire project has been developed from the leacture slides of Dr. Mitesh Khapra, Indian Institute of Technology Madras: http://cse.iitm.ac.in/~miteshk/CS6910.html#schedule
2. http://www.cs.cmu.edu/~arielpro/15381f16/c_slides/781f16-17.pdf 
3. https://arxiv.org/pdf/1711.05101.pdf
4. https://wandb.ai
5. https://github.com/
6. https://stats.stackexchange.com/questions/153605
7. https://cs231n.github.io/neural-networks-3/#annealing-the-learning-rate

