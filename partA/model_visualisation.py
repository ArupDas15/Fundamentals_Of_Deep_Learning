import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from keras.utils import to_categorical
from partA.data import preprocess_img

def report_accuracy():
	best_model = keras.models.load_model('best_model')
	best_model.summary()
	test = preprocess_img(datatype='test',batch_size=32,target_size=(240,240))
	# Reference: https://stackoverflow.com/questions/61742556/valueerror-shapes-none-1-and-none-2-are-incompatible
	one_hot_label = to_categorical(test.y)
	""" Evaluates the mean loss for a batch of inputs and accuracy.
		If the model has lower loss (score) at test time, it will have lower prediction error and hence higher the accuracy. 
		Similarly, when test accuracy is low the score is higher.
	"""
	score, acc = best_model.evaluate(test.x,one_hot_label, batch_size=32, verbose = 0)
	print('Test accuracy:', acc)

report_accuracy()


