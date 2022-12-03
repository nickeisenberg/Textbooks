import re
import string
from tensorflow.keras.layers import TextVectorization
import tensorflow as tf


def custom_standardization_fn(string_tensor):
    lowercase_string = tf.strings.lower(string_tensor)
    return tf.strings.regex_replace(
            lowercase_string, f'[{string.punctuation}]', '')

x = [string.punctuation]
print(x)
x0 = [re.escape(string.punctuation)]
print(x0)

string_ex = 'this is !!a string_#'
new_string = tf.strings.regex_replace(string_ex, '[!_#]', '')
print(new_string)

def custom_split_fn(string_tensor):
    return tf.strings.split(string_tensor)


text_vecotrization = TextVectorization(
        output_mode='int',
        standardize=custom_standardization_fn,
        split=custom_split_fn,
        )

dataset = ['I write, erase, rewrite',
           'Erase again, and then',
           'A poppy blooms.']
text_vecotrization.adapt(dataset)

text_vecotrization.get_vocabulary()

vocabulary = text_vecotrization.get_vocabulary()
print(vocabulary)

test_sentance = 'I write, rewrite, and still write again'

encoded_sentance = text_vecotrization(test_sentance)
print(encoded_sentance)

inverse_vocab = dict(enumerate(vocabulary))
print(inverse_vocab)

decoded_sentance = ' '.join([inverse_vocab[int(i)] for i in encoded_sentance])
print(decoded_sentance)
