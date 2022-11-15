from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sys import exit

path_to_model = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter8/'
model_name = 'convnet_from_scratch_with_aug.keras'

model = keras.models.load_model(path_to_model + model_name)

# print(model.summary())

img_path = keras.utils.get_file(
    fname='cat.jpg',
    origin='https://img-datasets.s3.amazonaws.com/cat.jpg')


def get_img_array(img_path, target_size):
    img = keras.utils.load_img(
        img_path,
        target_size=target_size)
    array = keras.utils.img_to_array(img)
    array = np.expand_dims(array, axis=0)
    return array


img_tensor = get_img_array(img_path, target_size=(180, 180))

# plt.imshow(img_tensor[0].astype('uint8'))
# plt.show()

layer_outputs = []
layer_names = []

for layer in model.layers:
    if isinstance(layer, (layers.Conv2D, layers.MaxPooling2D)):
        layer_names.append(layer.name)
        layer_outputs.append(layer.output)

activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

# lets visualize all the channels for each output

images_per_row = 16
fig, ax = plt.subplots(3,3)
ax = ax.reshape(-1)
count = 0
for layer_name, layer_activation in zip(layer_names, activations):
    size = layer_activation.shape[1]
    channels = layer_activation.shape[-1]
    col_len = channels // images_per_row
    display_grid = np.zeros(
        ((size + 1) * col_len - 1, (size + 1) * images_per_row - 1)
    )

    for col in range(col_len):
        for row in range(images_per_row):
            channel_index = col * images_per_row + row
            channel_image = layer_activation[0, :, :, channel_index].copy()
            if np.sum(channel_image) != 0:
                channel_image -= np.mean(channel_image)
                channel_image /= np.std(channel_image)
                channel_image *= 64
                channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[
                col * (size + 1): col * (size + 1) + size,
                row * (size + 1): row * (size + 1) + size
            ] = channel_image
    scale = 1. / size
    # plt.figure(figsize=(scale * display_grid.shape[1],
    #                     scale * display_grid.shape[0]))
    # plt.title(layer_name)
    # plt.grid(False)
    # plt.axis('off')
    # plt.imshow(display_grid, aspect='auto', cmap='viridis')
    ax[count].set_title(layer_name)
    ax[count].grid(False)
    ax[count].axis('off')
    ax[count].imshow(display_grid, aspect='auto', cmap='viridis')
    count += 1

plt.suptitle('layer activations of the input')
plt.show()

