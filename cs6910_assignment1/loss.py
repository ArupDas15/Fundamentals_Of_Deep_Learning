"""This file will contain various methods for calculation of loss functions"""
import numpy as np


# calculate cross entropy
def cross_entropy(label, softmax_output):
    # as we have only one true label, we have simplified the function for faster calculation.
    if softmax_output[label] < 10 ** -8:
        return -np.log(10 ** -8)
    return -np.log(softmax_output[label])


def squared_error(label, softmax_output):
    true_vector = np.zeros_like(softmax_output)
    true_vector[label] = 1
    size = float(len(softmax_output))
    return np.array([(np.linalg.norm(true_vector - softmax_output) ** 2) / size]).reshape((1, 1))
