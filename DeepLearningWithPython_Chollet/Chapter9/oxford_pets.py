from sys import exit
import os
import matplotlib.pyplot as plt
from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import numpy as np
import random
from tensorflow import keras
from tensorflow.keras import layers

inp_dir = '/Users/nickeisenberg/GitRepos/DataSets_local/Oxford_IIIT_Pets/images/'
target_dir = '/Users/nickeisenberg/GitRepos/DataSets_local/Oxford_IIIT_Pets/annotations/trimaps/'

input_img_paths = sorted(
    [os.path.join(inp_dir, fname)
     for fname in os.listdir(inp_dir)
     if fname.endswith('.jpg')])

target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith('.png')])

# plt.imshow(load_img(input_img_paths[9]))
# plt.show()


def display_target(target_array):
    normalized_array = (target_array.astype('uint8') - 1) * 127
    plt.axis('off')
    plt.imshow(normalized_array[:, :, 0])


# img = img_to_array(load_img(target_paths[9], color_mode='grayscale'))
# display_target(img)
#
# plt.show()

# load the inputs and targets into arrays
img_size = (200, 200)  # arbitarty choice
num_imgs = len(input_img_paths)

random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_paths)


def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode='grayscale'))
    img = img.astype('uint8') - 1
    return img


input_imgs = np.zeros((num_imgs,) + (img_size) + (3,), dtype='float32')
targets = np.zeros((num_imgs,) + (img_size) + (1,), dtype='uint8')
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

# split to training and validation
num_val_samples = 1000
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]


# define the model
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same')(x)

    x = layers.Conv2DTranspose(256, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(256, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

    model = keras.Model(inputs, outputs)

    return model


model = get_model(img_size=img_size, num_classes=3)

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='oxford_segmentation.keras',
        save_best_only=True,
        monitor='val_loss')  # this is default
]

history = model.fit(train_input_imgs, train_targets,
                    epochs=30,
                    batch_size=64,
                    validation_data=(val_input_imgs, val_targets),
                    callbacks=callbacks)

epochs = range(1, len(history.history['loss']) + 1)
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.figure()
plt.plot(epochs, loss, 'bo', label='training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.legend()
plt.title('training and validation loss: image segmentation')
plt.show()

model = keras.models.load_model('oxford_segmentation.keras')

i = 4
test_image = val_input_imgs[i]
plt.axis('off')
plt.imshow(array_to_img(test_image))

mask = model.predict(test_image.reshape((1,) + test_image.shape))[0]


def display_mask(pred):
    mask = np.argmax(pred, axis=-1)
    mask *= 127
    plt.axis('off')
    plt.imshow()


display_mask(mask)
plt.show()
