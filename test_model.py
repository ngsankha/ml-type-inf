import json
import numpy as np
import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset
from tensorflow.keras.models import load_model

MODEL_TYPE = "names"
LABEL_CHOICE = "TOP"

## LOAD TOKENIZER    
with open('tokenizers/{}_tokenizer.pickle'.format(MODEL_TYPE), 'rb') as handle:
    lang_tokenizer = pickle.load(handle)

## LOAD LABELS
with open('labels/{}_{}_idx_to_label.pkl'.format(MODEL_TYPE, LABEL_CHOICE), 'rb') as f:    
    idx_to_label = pickle.load(f)
    
## LOAD SAVED MODEL
model = load_model('models/{}_{}_model.h5'.format(MODEL_TYPE, LABEL_CHOICE))

def run_model(inp):
    x = model.predict(lang_tokenizer.texts_to_sequences([inp]))
    print(idx_to_label[np.argmax(x)])


def run_twin_model(in1, in2):
    emb1, emb2 = tf.keras.preprocessing.sequence.pad_sequences(lang_tokenizer.texts_to_sequences([in1, in2]), maxlen = 82).squeeze()
    print(model.predict([[emb1], [emb2]]))
