import os
import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN, GRU, LSTM, Dense, Input, Embedding, Dropout, Activation, concatenate
from keras.layers import Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers, constraints, optimizers, layers
import pickle

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
embedding_file = 'glove.6B.300d.txt'

classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train_data[classes].values

train_sentences = train_data["comment_text"].fillna("fillna").str.lower()
test_sentences = test_data["comment_text"].fillna("fillna").str.lower()

max_features = 100000
max_len = 150
embed_size = 300

tokenizer = Tokenizer(max_features)
tokenizer.fit_on_texts(list(train_sentences))

tokenized_train_sentences = tokenizer.texts_to_sequences(train_sentences)
tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)

train_padding = pad_sequences(tokenized_train_sentences, max_len)
test_padding = pad_sequences(tokenized_test_sentences, max_len)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file, encoding='utf-8'))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

image_input = Input(shape=(max_len, ))
X = Embedding(max_features, embed_size, weights=[embedding_matrix])(image_input)
X = Bidirectional(GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(X)
# Dropout and R-Dropout sequence, inspired by Deep Learning with Python - Francois Chollet
avg_pl = GlobalAveragePooling1D()(X)
max_pl = GlobalMaxPooling1D()(X)
conc = concatenate([avg_pl, max_pl])
X = Dense(6, activation="sigmoid")(conc)
model = Model(inputs=image_input, outputs=X)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

saved_model = "Text_Model.hdf5"
checkpoint = ModelCheckpoint(saved_model, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]


batch_sz = 32
epoch = 2
model.fit(train_padding, y, batch_size=batch_sz, epochs=epoch, validation_split=0.1, callbacks=callbacks_list)

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)