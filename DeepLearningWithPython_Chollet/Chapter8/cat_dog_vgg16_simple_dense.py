from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt
import numpy as np

# Instantiating the vgg16 conv base
conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(180, 180, 3))


def get_all_features_and_labeles(dataset):
    all_features = []
    all_labels = []
    count = 1
    for images, labels in dataset:
        print(f'processing batch: {count} / {len(dataset)}')
        preprocessed_images = keras.applications.vgg16.preprocess_input(images)
        features = conv_base.predict(preprocessed_images)
        all_features.append(features)
        all_labels.append(labels)
        count += 1
    return np.concatenate(all_features), np.concatenate(all_labels)


path0 = '/Users/nickeisenberg/GitRepos/Textbooks/'
path1 = 'DeepLearningWithPython_Chollet/Chapter8/DataSets/dogs_vs_cats_small/'
path = path0 + path1

train_dataset = image_dataset_from_directory(
    path + 'train/',
    labels='inferred',  # this is default
    image_size=(180, 180),
    batch_size=32)
validation_dataset = image_dataset_from_directory(
    path + 'validation/',
    image_size=(180, 180),
    batch_size=32)
test_dataset = image_dataset_from_directory(
    path + 'test/',
    image_size=(180, 180),
    batch_size=32)

train_features, train_labels = get_all_features_and_labeles(train_dataset)
val_features, val_labels = get_all_features_and_labeles(validation_dataset)
test_features, test_labels = get_all_features_and_labeles(test_dataset)

# define the dense classifer
inputs = keras.Input(shape=(5, 5, 512))
x = layers.Flatten()(inputs)
x = layers.Dense(256)(x)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='feature_extraction.keras',
        save_best_only=True,
        monitor='val_loss')
]

history = model.fit(
    train_features, train_labels,
    epochs=20,
    validation_data=(val_features, val_labels),
    callbacks=callbacks)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.legend()
plt.title('Accuracy of the dense model with preprocessed vgg16 images')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.title('Loss of the dense model with preprocessed vgg16 images')

plt.show()
