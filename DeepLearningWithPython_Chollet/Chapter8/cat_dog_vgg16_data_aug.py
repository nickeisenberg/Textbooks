from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import image_dataset_from_directory
import matplotlib.pyplot as plt

conv_base = keras.applications.vgg16.VGG16(
    weights='imagenet',
    incude_top=False,)
conv_base.trainable = False

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip('horizontal'),
        layers.RandomRoation(.1),
        layers.RandomZoom(.2),
    ]
)

inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = keras.application.vgg16.preprocess_input(x)
x = conv_base(x)
x = layers.Flatten()(x)
x = layers.dense(256)(x)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.complile(loss='binary_crossentropy',
               optimizer='rmsprop',
               mertics=['accuracy'])

path0 = '/Users/nickeisenberg/GitRepos/Textbooks'
path1 = '/DeepLearningWithPython_Chollet/Chapter8/DataSets/dogs_vs_cats_small/'
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

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='feature_extraction_with_data_aug.keras',
        save_best_only=True,
        monitor='val_loss')
 ]

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=callbacks)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.legend()
plt.title('Training and validation accuracy')

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.legend()
plt.title('Training and validation loss')

plt.show()

test_model = keras.models.load_model(
    'feature_extraction_with_data_aug.keras')
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test loss: {test_loss}\nTest Accuracy: {test_acc}')

# We can continue and find tune the last three layers of conv_base
conv_base.trainable = True
for layer in conv_base[:-4]:
    layer.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-5),
              metrics=['accuracy'])

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath='fine_tuning.keras',
        save_best_only=True,
        monitor='val_loss')
]

history = model.fit(
    train_dataset,
    epochs=30,
    validation_data=validation_dataset,
    callbacks=callbacks)

model = keras.models.load_model('fine_tuning.keras')
test_loss, test_acc = model.evaluate(test_dataset)
print(f'test loss: {test_loss}\ntest_acc: {test_acc}')
