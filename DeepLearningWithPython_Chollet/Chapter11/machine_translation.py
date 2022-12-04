import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import string
import re

path_to_datasets = '/Users/nickeisenberg/GitRepos/DataSets_local/'
dataset_dir = 'spa-eng/'
f_name = 'spa.txt'
full_path = path_to_datasets + dataset_dir + f_name

with open(full_path) as f:
    count = 1
    for line in f:
        print(line)
        if count == 6:
            break
        count += 1

with open(full_path) as f:
    lines = f.read().split('\n')[:-1]
text_pairs = []
for line in lines:
    english, spanish = line.split('\t')
    spanish = '[start]' + spanish + '[end]'
    text_pairs.append((english, spanish))

print(random.choice(text_pairs))

random.shuffle(text_pairs)

num_val_samples = int(.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[: num_train_samples]
val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
test_pairs = text_pairs[num_val_samples + num_train_samples:]

strip_chars = string.punctuation + '¿'
strip_chars = strip_chars.replace('[', '')
strip_chars = strip_chars.replace(']', '')

def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(
            lowercase, f'[{re.escape(strip_chars)}]', '')

vocab_size = 15000
sequence_length = 20

source_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length,
        )

target_vectorization = layers.TextVectorization(
        max_tokens=vocab_size,
        output_mode='int',
        output_sequence_length=sequence_length + 1,
        standardize=custom_standardization)

train_english_texts = [pair[0] for pair in text_pairs]
train_spanish_texts = [pair[1] for pair in text_pairs]
source_vectorization.adapt(train_english_texts)
target_vectorization.adapt(train_spanish_texts)

batch_size = 64
def format_dataset(eng, spa):
    eng = source_vectorization(eng)
    spa = target_vectorization(spa)
    return({
        'english': eng,
        'spanish': spa[:, :-1],
        }, spa[:, 1:])

def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset, num_parallel_calls=4)
    return dataset.shuffle(2048).prefetch(16).cache()

train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f"inputs['english'].shape: {inputs['english'].shape}")
    print(f"inputs['spanish'].shape: {inputs['spanish'].shape}")
    print(f'targets.shape: {targets.shape}')



