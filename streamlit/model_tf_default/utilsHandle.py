import os
import sys
from pathlib import Path
from os.path import join
from os import listdir
import tensorflow as tf
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

max_length, attention_features_shape = 50, 64

# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
# Shape of the vector extracted from InceptionV3 is (64, 2048)
# These two variables represent that vector shape
features_shape = 2048
attention_features_shape = 64

# Create mappings for words to indices and indices to words.
import pickle

with open("streamlit/model_tf_default/utils/get_vocabulary.txt", 'r', encoding="utf8") as f:
    vocabulary = [line.rstrip('\n') for line in f]

word_to_index = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token="",
    vocabulary=vocabulary)
index_to_word = tf.keras.layers.experimental.preprocessing.StringLookup(
    mask_token="",
    vocabulary=vocabulary,
    invert=True)


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # attention_hidden_layer shape == (batch_size, 64, units)
        attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                             self.W2(hidden_with_time_axis)))

        # score shape == (batch_size, 64, 1)
        # This gives you an unnormalized score for each image feature.
        score = self.V(attention_hidden_layer)

        # attention_weights shape == (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


encoder_load = CNN_Encoder(embedding_dim)
# decoder_load = RNN_Decoder(embedding_dim, units, new_v.vocabulary_size())
decoder_load = RNN_Decoder(embedding_dim, units, 1331)

image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
# th??? b??? 2 layer cu???i nh?? trong ytb

image_features_extract_model_load = tf.keras.Model(new_input, hidden_layer)

# Restore the weights
encoder_load.load_weights('streamlit/model_tf_default/utils/CNN_Encoder_weight')
decoder_load.load_weights('streamlit/model_tf_default/utils/RNN_Decoder_weight')
image_features_extract_model_load.load_weights('streamlit/model_tf_default/utils/image_features_extract_model_weight')


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.experimental.preprocessing.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def evaluate_load(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder_load.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model_load(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder_load(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)

    result = ""

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder_load(dec_input,
                                                              features,
                                                              hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        predicted_word = tf.compat.as_text(index_to_word(tf.constant(predicted_id)).numpy())

        if predicted_word == '<end>':
            return result, attention_plot

        result += predicted_word
        result += " "

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def predict(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder_load.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model_load(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder_load(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index('<start>')], 0)

    result = ""

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder_load(dec_input,
                                                              features,
                                                              hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()

        predicted_word = tf.compat.as_text(index_to_word(tf.constant(predicted_id)).numpy())

        if predicted_word == '<end>':
            return result

        result += predicted_word
        result += " "

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result
