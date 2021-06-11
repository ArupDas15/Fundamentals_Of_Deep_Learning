import numpy as np


def cross_entropy_grad(y_hat, label):
    # grad w.r.t out activation
    temp = np.zeros_like(y_hat)
    # If the initial guess is very wrong. This gradient will explode. This places a limit on that.
    if y_hat[label] < 10 ** -8:
        y_hat[label] = 10 ** -8
    temp[label] = -1 / (y_hat[label])
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp


def squared_error_grad(y_hat, label):
    # grad w.r.t out activation
    temp = np.copy(y_hat)
    temp[label] -= 1
    temp = 2 * temp
    temp = temp / len(y_hat)
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp


def output_grad(y_hat, label, loss_type):
    if loss_type == 'cross_entropy':
        return cross_entropy_grad(y_hat=y_hat, label=label)
    elif loss_type == 'squared_error':
        return squared_error_grad(y_hat=y_hat, label=label)


def last_grad(y_hat, label):
    # grad w.r.t out last layer
    temp = np.copy(y_hat)
    temp[label] = temp[label] - 1
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp


# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is sigmoid.We have passed h_is
def sigmoid_grad(post_activation):
    return np.multiply(post_activation, 1 - post_activation)


# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is tanh. We have passed h_is
def tanh_grad(post_activation):
    return 1 - np.power(post_activation, 2)


# this function helps in calculation of gradient w.r.t 'a_i''s when activation function is relu.
def relu_grad(pre_activation_vector):
    grad = np.copy(pre_activation_vector)
    # making +ve and 0 component 1
    grad[grad >= 0] = 1
    # making -ve component 0
    grad[grad < 0] = 0
    return grad


def a_grad(network, transient_gradient, layer):
    # grad w.r.t  a_i's layer
    if network[layer]['context'] == 'sigmoid':
        active_grad_ = sigmoid_grad(network[layer]['h'])
    elif network[layer]['context'] == 'tanh':
        active_grad_ = tanh_grad(network[layer]['h'])
    elif network[layer]['context'] == 'relu':
        active_grad_ = relu_grad(network[layer]['a'])
    temp = np.multiply(transient_gradient[layer]['h'], active_grad_)
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp
    # hadamard multiplication


def h_grad(network, transient_gradient, layer):
    # grad w.r.t out h_i layer
    network[layer]['weight'].transpose()
    temp = network[layer + 1]['weight'].transpose() @ transient_gradient[layer + 1]['a']
    norm = np.linalg.norm(temp)
    if norm > 100.0:
        return temp * 100.0 / norm
    else:
        return temp


def w_grad(network, transient_gradient, layer, x):
    if layer == 0:
        temp = transient_gradient[layer]['a'] @ x.transpose()
    else:
        temp = transient_gradient[layer]['a'] @ network[layer - 1]['h'].transpose()
    norm = np.linalg.norm(temp)
    if norm > 10000.0:
        return temp * 10000.0 / norm
    else:
        return temp
