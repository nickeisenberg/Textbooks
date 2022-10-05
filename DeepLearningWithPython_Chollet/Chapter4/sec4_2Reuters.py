# Listing 4.11 Loading Reuters dataset 
from tensorflow.keras.datasets import reuters
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Listing 4.12 Decoding the newswires 
word_index = reuters.get_word_index()
reverse_word_index = dict(
    [(value, key) for (key, value) in word_index.items()])
decoded_newswire = ' '.join(
    [reverse_word_index.get(i-3, '?') for i in train_data[0]])

# Listing 4.13 Encoding the input data 
import numpy as np
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, seq in enumerate(sequences):
        for val in seq:
            results[i, val] = 1
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# Listing 4.14 Encoding the labels (categorical encoding)
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for key, label in enumerate(labels):
        results[key, label] = 1
    return results
y_train = to_one_hot(train_labels)
y_test = to_one_hot(test_labels)

# Listing 4.15 Model definition
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(46, activation='softmax')
])

# Listing 4.16 Compiling the model 
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Listing 4.17 Setting aside a validation set
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

# Listing 4.18 Training the model 
history = model.fit(partial_x_train,
          partial_y_train,
          epochs=20,
          batch_size=512,
          validation_data=(x_val, y_val))

# Listing 4.19 Plotting the training and validation loss
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = len(loss)
plt.plot(range(1, epochs + 1), loss, 'bo', label='Training Loss')
plt.plot(range(1, epochs + 1), val_loss, 'b', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Listing 4.20 Plotting the training and validation accuracy 
plt.clf()
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
epochs = len(loss)
plt.plot(range(1, epochs + 1), accuracy, 'bo', label='Training Accuracy')
plt.plot(range(1, epochs + 1), val_accuracy, 'b', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# 4.2.5 Generating predicitons on new data 
predictions = model.predict(x_test)
print(np.argmax(predictions[0]))
print(test_labels[0])
