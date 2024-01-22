import tensorflow as tf
import einops
from my_mixer import MLPMixer
cifar = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.astype("float32"), x_test.astype("float32")

model= MLPMixer(input_shape=x_train.shape[1:],
                num_classes=10, 
                num_blocks=8, 
                patch_size=8,
                hidden_dim=512, 
                tokens_mlp_dim=2048,
                channels_mlp_dim=256,
)

# model = tf.keras.applications.ResNet50(
#     include_top=True,
#     weights=None,
#     input_shape=(32, 32, 3),
#     classes=10,)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
model.build(input_shape=(None, 32, 32, 3))
model.summary()
model.fit(x_train, y_train, epochs=30, batch_size=512)

print('done')