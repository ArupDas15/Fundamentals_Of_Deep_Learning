import pickle
from keras.datasets import fashion_mnist
from sklearn import metrics
import numpy as np
import math
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import wandb
import os

# load the trained network
filename_model = 'neural_network.object'
network = pickle.load(open(filename_model, 'rb'))


(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

def softmax(pre_activation_vector):
    post_act = np.copy(pre_activation_vector)
    # we are shifting the value of exponent because in case of large error, there can be nan problem,this is the fix
    max_exponent = np.max(post_act)
    post_act = np.exp(post_act - max_exponent)
    post_act = post_act / np.sum(post_act)
    return post_act

# this function calculates softmax
def relu(pre_activation_vector):
    post_act = np.copy(pre_activation_vector)
    # get the position of vector that is -ve and make them 0
    post_act[post_act < 0] = 0
    return post_act

# this function helps in calculation of sigmoid function value of a component of vector
def sigmoid_element_wise(vector_component):
    # if-else to prevent math overflow
    if vector_component >= 0:
        return 1 / (1 + math.exp(-vector_component))
    else:
        return math.exp(vector_component) / (math.exp(vector_component) + 1)

# this function calculated sigmoid of pre - activation layer
def sigmoid(pre_activation_vector):
    # create a vector of same shape as input
    activated_vector = np.empty_like(pre_activation_vector)
    # iterate over input
    for i, elem in np.ndenumerate(pre_activation_vector):
        # calculate component wise sigmoid
        activated_vector[i] = sigmoid_element_wise(elem)
    return activated_vector


# this function handles the input and redirects the request to proper function
def activation_function(pre_activation_vector, context):
    if context == 'softmax':
        # calling softmax
        return softmax(pre_activation_vector)
    elif context == 'sigmoid':
        # calling sigmoid
        return sigmoid(pre_activation_vector)
    elif context == 'tanh':
        # creating tanh
        return np.copy(np.tanh(pre_activation_vector))
    elif context == 'relu':
        # calling relu
        return relu(pre_activation_vector)
    else:
        # Error handling
        return None

def forward_propagation(n, x):
    for i in range(n):
        if i == 0:
            network[i]['a'] = network[i]['weight'] @ x + network[i]['bias']
        else:
            network[i]['a'] = network[i]['weight'] @ network[i - 1]['h'] + network[i]['bias']

        network[i]['h'] = activation_function(network[i]['a'], context=network[i]['context'])


# Reference: https://github.com/zalandoresearch/fashion-mnist/blob/master/README.md#Labels
cm_plot_labels = ['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
			   'Ankle boot']

def predict_label(number_of_layer):
	acc = 0
	y_pred=[]
	for x, y in zip(testX, testy):
		forward_propagation(number_of_layer, x.reshape(784, 1) / 255.0)
		max_prob = np.argmax(network[number_of_layer - 1]['h'])
		if max_prob == y:
			acc += 1
		y_pred.append(max_prob)
	print("Accuracy: ",str((acc/len(testy))*100),"%")
	cm=metrics.confusion_matrix(y_true=testy,y_pred=y_pred)
	df_cm = pd.DataFrame(cm, index=[i for i in cm_plot_labels],
						 columns=[i for i in cm_plot_labels])
	print(df_cm)
	plt.figure(figsize=(10, 10))
	ax = sn.heatmap(df_cm, annot=True,  cmap='Blues', fmt='d',linewidths=3, linecolor='black')
	ax.set_yticklabels(cm_plot_labels,rotation=0)
	plt.xlabel("True Class")  # x-axis label
	plt.ylabel("Predicted Class")  # y-axis label
	plt.title('Confusion Matrix of FASHION-MNIST Dataset', fontsize=20)
	plt.show()


predict_label(len(network))


