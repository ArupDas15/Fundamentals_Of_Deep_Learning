import numpy as np
import math


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


# this function calculates softmax
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
