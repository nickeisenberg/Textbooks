import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import text_dataset_from_directory

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


class TransformerEncoder(layers.Layer):

    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim,
        self.dense_dim = dense_dim,
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAtttention(
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
        attention_output = self.attention(
                inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'dense_dim': self.dense_dim,
            })
        return config


vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32

inputs = keras.Input(shape=(None,), dtype='int64')
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(.5)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

cback_path = '/Users/nickeisenberg/GitRepos/Textbooks/DeepLearningWithPython_Chollet/Chapter11/'
c_back_name = 'transforer_encoder.keras'
callbacks = [
        keras.callbacks.ModelCheckpoint(
            cback_path + c_back_name,
            save_best_only=True)]

model.fit(int_train_ds,
          validation_data=int_val_ds,
          epochs=20,
          callbacks=callbacks)

model = keras.models.load_model(
        cback_path + c_back_name,
        custom_objects={'TransformerEncoder': TransformerEncoder})
print(f'Model accuracy: {model.evaluate(int_test_ds):.3f}')
