import os, pathlib, shutil, random
from tensorflow import keras
from tensorflow.keras import layers

# Preparing the train, val and test datasets
offline_dir = '/Users/nickeisenberg/GitRepos/DataSets_local'
base_dir = offline_dir / pathlib.Path('aclImdb')

val_dir = base_dir / 'val'
train_dir = base_dir / 'train'
for category in ['neg', 'pos']:
    os.makedirs(val_dir / category)
    files = os.listdir(train_dir / category)
    random.Random(1337).shuffle(files)
    num_val_samples = int(.2 * len(files))
    val_files = files[-num_val_samples:]
    for fname in val_files:
        shutil.move(train_dir / category / fname,
                    val_dir / category / fname)

batch_size = 32

train_ds = keras.utils.text_dataset_from_directory(
        base_dir / 'train',
        batch_size=batch_size)

val_ds = keras.utils.text_dataset_from_directory(
        base_dir / 'val',
        batch_size=batch_size)

test_ds = keras.utils.text_dataset_from_directory(
        base_dir / 'test',
        batch_size=batch_size)

for inps, tars in train_ds:
    print(f'input shape: {inps.shape}')
    print(f'input dtype: {type(inps)}')
    print(f'target shape: {tars.shape}')
    break

# vecotrizing the data into 1 gram batch of words
text_vectorization = layers.TextVectorization(
        max_tokens=20000,
        output_mode='multi_hot')

text_only_train_ds = train_ds.map(lambda x, y: x)

text_vectorization.adapt(text_only_train_ds)

binary_1gram_train_ds = train_ds.map(lambda x, y: (text_vectorization(x), y),
                                     num_parallel_calls=4)

binary_1gram_val_ds = val_ds.map(lambda x, y: (text_vectorization(x), y),
                                 num_parallel_calls=4)

binary_1gram_test_ds = test_ds.map(lambda x, y: (text_vectorization(x), y),
                                   num_parallel_calls=4)

for inps, tars in binary_1gram_train_ds:
    print(f'Input shape: {inps.shape}')
    print(f'Target shape: {tars.shape}')
    print(inps[0])
    print(tars[0])
    break

# general model
def get_model(max_tokens=20000, hidden_dim=3):
    inputs = keras.Input(shape=(max_tokens,))
    x = layers.Dense(hidden_dim, activation='relu')(inputs)
    x = layers.Dropout(.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = get_model()
print(model.summary())

# fitting the model to the 1-gram batch of words and testing the accuracy
w_dir = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter11'
save_path = w_dir + '/binary_1gram.keras'
print(save_path)

callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path,
            save_best_only=True)]

model.fit(binary_1gram_train_ds.cache(),
          validation_data=binary_1gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model(save_path)
print(f'Model accuracy: {model.evaluate(binary_1gram_test_ds)[1]:.3f}')

# vectorizing the datasets into 2-gram batch of words
text_vectorization = TextVectorization(
        ngrams=2,
        max_tokens=20000,
        output_mode='multi_hot')

text_vectorization.adapt(text_only_train_ds)

binary_2gram_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

binary_2gram_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

binary_2gram_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

model = get_model()

# training and testing a model on the 2-gram batch of words datasets
save_path = w_dir + '/binary_2gram.keras'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path,
            save_best_only=True)]

model.fit(binary_2gram_train_ds.cache(),
          validation_data=binary_2gram_val_ds.cache(),
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model(save_path)
print(f'Model accuracy: {model.evaluate(binary_2gram_test_ds)[1]:.3f}')

#  creating 2-gram batch of word datasets with the tf_idf output
text_vectorization = TextVectorization(
        ngrams=2,
        max_tokens=20000,
        output_mode='tf_idf')

text_vectorization.adapt(text_only_train_ds)

tfidf_2gram_train_ds = train_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

tfidf_2gram_val_ds = val_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

tfidf_2gram_test_ds = test_ds.map(
        lambda x, y: (text_vectorization(x), y),
        num_parallel_calls=4)

model = get_model()

# fitting the model and testing fo accuracy
save_path = w_dir + '/tfidf_2gram.keras'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            save_path,
            save_best_only=True)]

model.fit(tfidf_2gram_train_ds,
          validation_data=tfidf_2gram_val_ds,
          epochs=10,
          callbacks=callbacks)

model = keras.models.load_model(save_path)
print(f'model accuracy: {model.evaluate(tfidf_2gram_test_ds)[1]:.3f}')

