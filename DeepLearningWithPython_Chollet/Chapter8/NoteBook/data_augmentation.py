import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras import Sequential, layers

path = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter8/DataSets/dogs_vs_cats_small/'

train = image_dataset_from_directory(
    path + 'train/',
    image_size=(180,180),
    shuffle=False,
    batch_size=32)

data_augmentaion = Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.2),
    ]
)

fig, axs = plt.subplots(3,3)
axs = axs.reshape(-1)
for ax in axs:
    for images, _ in train.take(1):
        aug_images = data_augmentaion(images)
        ax.imshow(np.array(aug_images[0], dtype='uint8'))
        ax.axis('off')

plt.show()
