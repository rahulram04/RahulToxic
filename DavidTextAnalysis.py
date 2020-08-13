import pandas as pd
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

data = pd.read_csv('Messages.csv', sep=';')
df = data[(data.sender_name == 'rahulrram04@gmail.com')].reset_index(drop=True)
test_sentences = df["message"].fillna("fillna").str.lower()

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

tokenized_test_sentences = tokenizer.texts_to_sequences(test_sentences)
test_padding = pad_sequences(tokenized_test_sentences, 150)

model = load_model("Text_Model.hdf5")
model.get_weights()
model.optimizer

preds = model.predict(test_padding)
df['Preds'] = list(map(sum, preds))

df.to_csv('TextToxicity.csv')
