import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import text_dataset_from_directory

path_to_IMDB = '/Users/nickeisenberg/GitRepos/DataSets_local/aclImdb/'

batch_size = 32
train_ds = text_dataset_from_directory(path_to_IMDB + 'train',
                                       batch_size=batch_size)
val_ds = text_dataset_from_directory(path_to_IMDB + 'val',
                                       batch_size=batch_size)
test_ds = text_dataset_from_directory(path_to_IMDB + 'test',
                                       batch_size=batch_size)
text_only_train_ds = train_ds.map(
        lambda x, y: x)

max_tokens = 20000
max_length = 600
text_vectorization = layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode='int',
        output_sequence_length=max_length)
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

glove_path = '/Users/nickeisenberg/GitRepos/DataSets_local/Glove_6B/'
glove_file = 'glove.6B.100d.txt'
path_to_glove_file = glove_path + glove_file

embeddings_index = {}
with open(path_to_glove_file) as f:
    for line in f:
        count += 1
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, 'f', sep=' ')
        embeddings_index[word] = coefs

print(f'Found {len(embeddings_index)} word vectors.')

embeddings_dim = 100

vocabulary =  text_vectorization.get_vocabulary()
word_index = dict(zip(vocabulary, range(len(vocabulary))))

embeddings_matrix = np.zeros((max_tokens, embeddings_dim))
for word, i in word_index.items():
    if i < max_tokens:
        embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector

embedding_layer = layers.Embedding(
        max_tokens,
        embeddings_dim,
        embeddings_initializer=keras.initializers.Constant(embeddings_matrix),
        trainable=False,
        mask_zero=True)

inputs = keras.Input(shape=(None,), dtype='int64')
embedded = embedding_layer(inputs)
x = layers.Bidirectional(layers.LSTM(32))(embedded)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

c_back_path = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter11/'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            c_back_path + 'glove_embeddings_seq_model.keras',
            save_best_only=True)]

model.fit(int_train_ds,
          validation_data=int_val_ds,
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model(c_back_path +
                                'glove_embeddings_seq_model.keras')
print(f'Model accuracy: {model.evaluate(int_test_ds):.3f}')

