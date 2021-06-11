from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from mpl_toolkits.axes_grid1 import ImageGrid
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


model=tf.keras.models.load_model('/content/drive/MyDrive/Models/model_final.keras')
path = '/content/iNaturalist_Dataset/inaturalist_12K/test/Plantae/04bf64c3ad09e7eb64f0f7f86316a547.jpg'
loaded_image = image.load_img(path, target_size=(400, 400))
img_numpy = image.img_to_array(loaded_image)
img_numpy = np.expand_dims(img_numpy, axis=0)
'''
Custom gradient
'''

@tf.custom_gradient
def guidedRelu(input):
  #gradient calculation changes
  def grad(gradient):
    return tf.cast(gradient>0,"float32") * tf.cast(input>0, "float32") * gradient
  #forward propagation stays the same
  return tf.nn.relu(input), grad
''' 
This method replaces all the relu activation function with guided relu defined above
'''
def replace_with_guided_relu_gradient():
    for layer in model.layers:
        if hasattr(layer, 'activation'):
            if layer.activation == tf.keras.activations.relu:
                layer.activation = guidedRelu
    model.compile()

def guided_backpropagation(neuron_number, conv_layer):
    model_input = model.input
    #choosing one neuron output
    neuron_output = tf.keras.activations.relu(model.get_layer(conv_layer).output[:,:,:,neuron_number])
    neuron = tf.keras.models.Model(inputs = [model.inputs],outputs = [neuron_output])
    with tf.GradientTape() as tape:
        # creating a tensor
        img_tensor=tf.cast(img_numpy,tf.float32)
        #keeping track of the tensor , so that it can be differentiated
        tape.watch(img_tensor)
        #mapping the function output , so that it can be differentiated
        output = neuron(img_tensor)
    return tape.gradient(output,img_tensor).numpy()[0]


#x = preprocess_input(x)

replace_with_guided_relu_gradient()
images=[]
'''
conv layers are named :-
conv2d-   1
conv2d_1- 2
conv2d_2- 3
conv2d_3- 4
conv2d_4- 5

 '''
for i in [10,11,12,22,23,25,21,32,39,103]:
  gradient = guided_backpropagation(i, 'conv2d_4')
  images.append(gradient)

figure = plt.figure(figsize=(15., 15.))
grid = ImageGrid(figure, 111,nrows_ncols=(2, 5),axes_pad=0.2)
for axes, img in zip(grid, images):
    axes.imshow(img)
    axes.axis('off')
plt.show()