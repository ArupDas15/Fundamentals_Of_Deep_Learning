from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import tensorflow as tf
import numpy as np
from keras.models import Model
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
testing=re_scale.flow_from_directory('/content/iNaturalist_Dataset/inaturalist_12K/validate', shuffle=True,target_size=(400,400),batch_size=1)
x_batch, y_batch = next(testing)
model = tf.keras.models.load_model('/content/drive/MyDrive/Models/model_final.keras')
layer_name = 'my_layer'
intermediate_layer_model = Model(inputs=model.input,outputs=model.layers[1].output)
intermediate_output = intermediate_layer_model.predict(x_batch)
images=[]
for i in range(128) :
    images.append(intermediate_output[0][:,:,i])
figure = plt.figure(figsize=(15, 15.))
grid = ImageGrid(figure, 111,nrows_ncols=(16, 8),axes_pad=0.1)
for axes, img in zip(grid, images):
    axes.imshow(img)
    axes.axis('off')
plt.show()

