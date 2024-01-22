import tensorflow as tf
from tensorflow.keras.layers import Add, Dense, Layer, LayerNormalization, GlobalAveragePooling2D, Conv2D, Permute, Softmax, Activation
import numpy as np 

class CreatePatches(Layer):
    def __init__(self, patch_size):
        super(CreatePatches, self).__init__()
        self.patch_size = patch_size

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        patches = tf.image.extract_patches(
            images=inputs,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        patch_dim = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dim])
        return patches
# sample_image = np.random.rand( 13 , 32 , 32 , 3 ) 
# layer = CreatePatches( 8 )
# print(layer( sample_image ).shape)

class MlpBlock(Layer):
    def __init__(self,dim = 256, hdim=512, name=None):
        super().__init__()
        self.dim = dim
        self.hdim = hdim
        self.dense1 = Dense(self.hdim)
        self.dense2 = Dense(self.dim,)
    
    def call(self, x):
        y = self.dense1(x)
        y = tf.nn.gelu(y)
        y = self.dense2(y)
        return y

class MixerBlock(Layer):
    def __init__(self, num_patches, channel_dim, tokens_mlp_dim, channels_mlp_dim, **kwargs):
        super(MixerBlock, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.channel_dim = channel_dim
        self.tokens_mlp_dim = tokens_mlp_dim
        self.channels_mlp_dim = channels_mlp_dim   

        self.norm1 = LayerNormalization(axis=1)
        self.permute1 = Permute((2,1))
        self.token_mixer = MlpBlock(num_patches, tokens_mlp_dim, name='token_mixer')

        self.permute2 = Permute((2,1))
        self.norm2 = LayerNormalization(axis=1)
        self.channel_mixer = MlpBlock(channel_dim, channels_mlp_dim, name='channel_mixer') 
        self.skip_connection = Add()
    
    def call(self,x):
        skip_x = x
        x = self.norm1(x)
        x = self.permute1(x)
        x = self.token_mixer(x)

        x = self.permute2(x)

        x = self.skip_connection([x, skip_x])
        skip_x = x  

        x =  self.norm2(x)  
        x = self.channel_mixer(x)   
        x = self.skip_connection([x, skip_x])

        return x
    
def MLPMixer(input_shape, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
    height, width, channels = input_shape
  
    num_patches = (height*width)//(patch_size**2)

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    print('input: ', x.shape)
    x = CreatePatches(patch_size)(x)
    print('cp: ', x.shape)
    x = Dense(hidden_dim)(x)
    # x = Conv2D(hidden_dim, kernel_size=(patch_size,patch_size), strides=(patch_size,patch_size), padding='same', name='projection')(x)
    # x = tf.keras.layers.Reshape([-1, hidden_dim]) (x)
    print(x.shape)
    for _ in range(num_blocks):
        x = MixerBlock(num_patches, hidden_dim, tokens_mlp_dim, channels_mlp_dim)(x)    

    # x = GlobalAveragePooling2D(x)
    x = LayerNormalization(name = 'pre_head_norm')(x)
    x = Dense(num_classes, activation='softmax', name='head')(x)
    x = tf.math.reduce_mean(x, axis=1)

    return tf.keras.Model(inputs=inputs, outputs=x)