import numpy as np
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

embed_dim = 256
latent_dim = 1024

source = keras.Input(shape=(None,), dtype='int64', name='english')
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(source)
encoded_source = layers.Bidirectional(
        layers.GRU(latent_dim), merge_mode='sum')(x)

past_target = keras.Input(shape=(None,), dtype='int64', name='spanish')
x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(past_target)
decoder_gru = layers.GRU(latent_dim, return_sequences=True)
x = decoder_gru(x, initial_state=encoded_source)
x = layers.Dropout(.5)(x)
target_next_step = layers.Dense(vocab_size, activation='softmax')(x)
seq2seq_rnn = keras.Model([source, past_target], target_next_step)

seq2seq_rnn.summary()

seq2seq_rnn.compile(optimizer='rmsprop',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

seq2seq_rnn.fit(train_ds, epochs=15, validation_data=val_ds)

# translate new sentences with the RNN encoder and decoder
spa_vocab = target_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20

def decode_sequence(input_sentence):
    tokenized_input_sentence = source_vectorization([input_sentence])
    decode_sentence = '[start]'
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = target_vectorization([decode_sentence])
        next_token_prediction = seq2seq_rnn.predict(
                [tokenized_input_sentence, tokenized_target_sentence])
        sampled_token_index = np.argmax(next_token_prediction[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += ' ' + sampled_token
        if sampled_token == '[end]':
            break
    return decode_sentence

# The transformer encoder
class TransforerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        slef.attention = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation='relu'),
                 layers.Dense(embed_dim), ]
                )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = slef.attention(
                inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = slef.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dense_dim': dense_dim,
            })
        return config

# The transformer decoder
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attenention_1 = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
        self.attenention_2 = layers.MultiHeadAttention(
                num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
                [layers.Dense(dense_dim, activation='relu'),
                 layers.Dense(embed_dim),]
                )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim,
            })
        return config

    def get_casual_attention(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype='int32')
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
                [tf.expand_dims(batch_size, -1),
                 tf.constant([1, 1], dtype='int32')], axis=0)
        return tf.tile(mask, mult)

    def call(self, inputs, encoder_outputs, mask=None):
        casual_mask = self.get_casual_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                    mask[:, tf.newaxis, :], dytpe='int32')
            attenention_output_1 = self.attenention_1(
                    query=inputs,
                    value=inputs,
                    key=inputs,
                    attention_mask=casual_mask)
            attenention_output_1 = self.layernorm_1(inputs + attenention_output_1)
            attenention_output_2 = self.attenention_2(
                    query=attenention_output_1,
                    value=encoder_outputs,
                    key=encoder_outputs,
                    attention_mask=padding_mask,
                    )
            attenention_output_2 = self.layernorm_2(
                    attenention_output_1 + attenention_output_2)
            proj_output = self.dense_proj(attenention_output_2)
            return self.layernorm_3(attenention_output_2 + proj_output)

# end to end transformer
embed_dim = 256
dense_dim = 2048
num_heads = 8

# PosistionalEmbedding from transformerIMDB.py. Need to put in here to run.
encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='english')
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransforerEncoder(embed_dim, dense_dim, num_heads)(x)

decoder_inputs = keras.Input(shape(None,), dtype='int64', name='spanish')
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, dense_dim, num_heads)(x, encoder_outputs)
x = layers.Dropout(.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation='softmax')(x)

transformer = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
transformer.compile(optimizer='rmsprop',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

transformer.fit(train_ds, validation_data=val_ds, epochs=30)

   

