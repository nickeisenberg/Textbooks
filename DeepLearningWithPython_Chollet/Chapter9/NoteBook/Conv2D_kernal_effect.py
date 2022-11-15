from tensorflow import keras
from tensorflow.keras import layers

inputs = keras.Input(shape=(32, 32, 3))
x = layers.Conv2D(32, 3, padding='same')(inputs)
x = layers.Conv2D(32, 3, padding='same')(x)
x = layers.Conv2D(32, 3)(x)
x = layers.Conv2D(32, 3, strides=2)(x)
x = layers.Conv2D(32, 1)(x)
x = layers.MaxPooling2D(2, padding='same')(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

print(model.summary())

