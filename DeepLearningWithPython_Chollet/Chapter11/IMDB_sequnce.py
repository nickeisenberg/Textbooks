import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


path_to_data = '/Users/nickeisenberg/GitRepos/DataSets_local/aclImdb/'

batch_size = 32
train_ds = keras.utils.text_dataset_from_directory(
        path_to_data + 'train',
        batch_size=batch_size)
val_ds = keras.utils.text_dataset_from_directory(
        path_to_data + 'val',
        batch_size=batch_size)
test_ds = keras.utils.text_dataset_from_directory(
        path_to_data + 'test',
        batch_size=batch_size)

max_tokens=20000
max_length=600
text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_length)

text_only_train_ds = train_ds.map(
    lambda x, y: x)

text_vectorization.adapt(text_only_train_ds)

int_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

int_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

int_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

for inps, tars in int_train_ds:
    print(inps.shape)
    print(tars.shape)
    print(tf.one_hot(inps[2], depth=max_tokens).shape)
    break

# create the model using one-hot encodding
inputs = keras.Input(shape=(None,), dtype='int64')
embedded = tf.one_hot(inputs, depth=max_tokens)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs,
                    outputs=outputs)

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print(model.summary())

save_path = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter11/'
model_name = 'one_hot_bidir_lstm.keras'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path + model_name,
            save_best_only=True)]

model.fit(int_test_ds,
          validation_data=int_val_ds,
          epochs=10,
          callbacks=callbacks)

# the model takes forever to fit and still is only 87% accurate.
model = keras.models.load_model(save_path)
print(f'model accuracy: {model.evaluate(int_test_ds):.3f}')

# using an embedding layer to handle the inputs and speed up performance
inputs = keras.Input(shape=(None,), dtype='int64')
embedded = layers.Embedding(input_dim=max_tokens, output_dim=256)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs,
                    outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_name = 'embeddings_bidir_gru.keras'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path + model_name,
            save_best_only=True)]

model.fit(int_train_ds,
          validation_data=int_val_ds,
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model(save_path + model_name)
print(f'model accuracy: {model.evaluate(int_test_ds):.3f}')

# similar model with the zeros masked
inputs = keras.Input(shape=(None,), dtype='int64')
embedded = layers.Embedding(input_dim=max_tokens,
                            output_dim=256,
                            mask_zero=True)(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs=inputs,
                    outputs=outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model_name = 'embeddings_bidir_gru_mask_zeros.keras'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path + model_name,
            save_best_only=True)]

model.fit(int_train_ds,
          validation_data=int_val_ds,
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model(save_path + model_name)
print(f'model accuracy: {model.evaluate(int_test_ds):.3f}')
