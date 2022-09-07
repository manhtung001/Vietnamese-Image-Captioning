import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.preprocessing as tfkp
import pickle
import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, \
    Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.sequence import pad_sequences

with open('streamlit/model_InceptionV3_LSTM/embedding_matrix.npy', 'rb') as f:
    embedding_matrix = np.load(f)

with open('streamlit/model_InceptionV3_LSTM/word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)

with open('streamlit/model_InceptionV3_LSTM/idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

# Get the InceptionV3 model trained on imagenet data
modelInceptionV3 = tfk.applications.inception_v3.InceptionV3(weights='imagenet')
# Remove the last layer (output softmax layer) from the inception v3
modelInceptionV3 = tfk.models.Model(modelInceptionV3.input, modelInceptionV3.layers[-2].output)


# we pass every image to this model to get the corresponding 2048 length feature vector
def preprocess(image_path):
    # Convert all the images to size 299x299 as expected by the inception v3 model
    img = tfkp.image.load_img(image_path, target_size=(299, 299))

    # Convert PIL image to numpy array of 3-dimensions
    x = tfk.utils.img_to_array(img)

    # Add one more dimension
    x = np.expand_dims(x, axis=0)

    # preprocess images using preprocess_input() from inception module
    x = tfk.applications.inception_v3.preprocess_input(x)

    return x

max_length = 35
vocab_size = 634
embedding_dim = 300

inputs1 = Input(shape=(2048,))
fe1 = Dropout(0.5)(inputs1)
fe2 = Dense(256, activation='relu')(fe1)
inputs2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
se2 = Dropout(0.5)(se1)
se3 = LSTM(256)(se2)
decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)
model = Model(inputs=[inputs1, inputs2], outputs=outputs)

model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

model.load_weights('streamlit/model_InceptionV3_LSTM/Info0_PhoW2V_syl_300___InceptionV39.h5')


# Function to encode a given image into a vector of size (2048, )
def encode_for_testing(image, model):
    image = preprocess(image)
    # Get the encoding vector for the image
    feature_vector = model.predict(image)
    # Reshape from (1, 2048) to (2048,)
    feature_vector = np.reshape(feature_vector, feature_vector.shape[1])
    return feature_vector


def greedySearch(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [word2idx[w] for w in in_text.split() if w in word2idx]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx2word[yhat]
        in_text += ' ' + word
        if word == '<end>':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final


def predict(path_img):
    image_tmp = encode_for_testing(path_img, modelInceptionV3).reshape(1, 2048)
    return greedySearch(image_tmp)



# pic = '/content/test.jpg'
# print(pic)
#
# x=plt.imread(pic)
# plt.imshow(x)
# plt.show()
# print("Greedy:",)
