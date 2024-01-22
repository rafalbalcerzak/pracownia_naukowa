from keras import backend as K
import tensorflow as tf
from my_mixer_noconv import MLPMixer
import numpy as np
from tensorflow.keras.layers import Input, Dense, Layer, LayerNormalization, GlobalAveragePooling2D, Conv2D
from einops.layers.tensorflow import Rearrange
import matplotlib.pyplot as plt

conv = Conv2D(128, kernel_size=(8,8), strides=(8,8), padding='same', name='projection')



test = np.zeros((32,32,3))[np.newaxis,...]
test[0,0:8,0:8,:] = 255
test[0,8:16,8:16,0] = 255
test[0,16:24,16:24,1] = 255
test[0,24:32,24:32,2] = 255
plt.imshow(test[0])
plt.show()

x = conv(test)
x = Rearrange("n h w c -> n (h w) c")(x)

print(x.shape)
plt.imshow(x[0])
plt.show()