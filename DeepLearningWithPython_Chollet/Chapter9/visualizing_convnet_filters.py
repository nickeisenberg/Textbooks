from sys import exit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

model = keras.applications.xception.Xception(
    weights='imagenet',
    include_top='False')

# for layer in model.layers:
#     if isinstance(layer, (layers.Conv2D, layers.SeparableConv2D)):
#         print(layer.name)

layer_name = 'block3_sepconv1'
layer = model.get_layer(name=layer_name)
feature_extractor = keras.Model(inputs=model.input, outputs=layer.output)

print(feature_extractor.summary())


def compute_loss(image, channel_index):
    activation = feature_extractor(image)
    filter_activation = activation[0, 2:-2, 2:-2, channel_index]
    return tf.reduce_mean(filter_activation)


@tf.function
def gradient_ascent_step(image, channel_index, learning_rate=10):
    with tf.GradientTape() as tape:
        tape.watch(image)
        loss = compute_loss(image, channel_index)
    grad = tape.gradient(loss, image)
    grad = tf.math.l2_normalize(grad)
    image += grad
    return image


img = tf.random.uniform(
    minval=.4,
    maxval=.6,
    shape=(1, 299, 299, 3))

for i in range(30):
    img = gradient_ascent_step(img, channel_index=2)

img = np.array(img)[0]
img -= img.mean()
img /= img.std()
img *= 64
img += 128
img = np.clip(img, 0, 255).astype('uint8')
img = img[25:-25, 25:-25, :]

plt.imshow(img)
plt.show()
