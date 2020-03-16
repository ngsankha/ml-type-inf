#!/usr/bin/env python
import json
from collections import Counter
import numpy as np

import tensorflow as tf
import tensorflow_text as text
import matplotlib.pyplot as plt
import pickle
from operator import itemgetter
import torch
from torch.utils.data.dataset import Dataset


## LABEL_CHOICE:
##   -"TOP" if picking the top most occuring labels in the dataset
##   -"PROG" if picking the labels occuring in at least MIN_PROGNUM_LABELS programs
LABEL_CHOICE = "PROG"

## Number of labels to pick from.
LABEL_NUM = 100

## When LABEL_CHOICE is "PROG", this is the minimum number of programs a type should occur
## in for it to be used as a label.
MIN_PROGNUM_LABELS = 5

def create_names_dataset(file_name):
    data_file = json.load(open(file_name, 'r'))
    # dataset will be a list of dicts of structure {'input': 'IDENTIFER_NAME', 'output': 'TYPE'}
    dataset = []
    ## dict mapping program names to a set of types observed in that program
    prog_type_dict = {} 
    for program,prog_vals in data_file.items():
        for class_key, class_val in prog_vals.items():
            for method, method_data in class_val.items():
                if method_data.get('return', None) and method_data['return'].get('type', None):
                    ## if we have a return type for this method, use the method's name as input,
                    ## and add the data to the dataset
                    ret_type = method_data['return']['type'].lower()
                    t = {'input': method, 'output': ret_type}
                    dataset.append(t)
                    prog_type_dict.setdefault(program, set()).add(ret_type)
                    for param_name,param_hash in method_data['params'].items():
                        ## for each parameter in the method data, add its data to the dataset
                        if param_hash.get('type', None):
                            param_type = param_hash['type'].lower()
                            t = {'input': param_name, 'output': param_type}
                            dataset.append(t)
                            prog_type_dict.setdefault(program, set()).add(param_type)

    ## below lines of code compute the count of how many programs each type appears in
    ## currently we don't use this count for anything, but it may be useful in the future
    ## for determining which types we want to predict.
    '''                    
    type_progcount = {}
    for typ in Counter([x['output'] for x in dataset]).keys():
    #TABprog_count = 0
    #TABfor prog_typs in prog_type_dict.values():
    #TAB    if typ in prog_typs:
    #TAB        prog_count += 1
    #TABtype_progcount[typ] = prog_count

    print(sorted(type_progcount.items(), key=itemgetter(1))[::-1])
    '''
                            
    return [dataset, prog_type_dict]

def create_comments_dataset(file_name):
    data_file = json.load(open(file_name, 'r'))
    # dataset will be a list of dicts of structure {'input': 'COMMENT', 'output': 'TYPE'}    
    dataset = []
    max_doc_size = 500 ## TEMPORARY TO CAP COMMENT SIZE
    prog_type_dict = {} ## dict to keep track of which programs feature which types
    for program,prog_vals in data_file.items():
        for class_key, class_val in prog_vals.items():
            for method, method_data in class_val.items():
                if method_data.get('return', None) and method_data['return'].get('type', None) and method_data['return'].get('doc', None):
                    ret_type = method_data['return']['type'].lower()
                    ret_doc = method_data['return']['doc']
                    t = {'input': ret_doc[:max_doc_size], 'output': ret_type}
                    dataset.append(t)
                    prog_type_dict.setdefault(program, set()).add(ret_type)
                for param_name,param_hash in method_data['params'].items():
                    if param_hash.get('type', None) and param_hash.get('doc', None):
                        param_type = param_hash['type'].lower()
                        param_doc = param_hash['doc'].lower()
                        t = {'input': param_doc[:max_doc_size], 'output': param_type}
                        dataset.append(t)
                        prog_type_dict.setdefault(program, set()).add(param_type)
    return [dataset, prog_type_dict]
                

## below lines of code compute the count of how many programs each type appears in
## currently we don't use this count for anything, but it may be useful in the future
## for determining which types we want to predict.
'''                    
type_progcount = {}
for typ in Counter([x['output'] for x in dataset]).keys():
    prog_count = 0
    for prog_typs in prog_type_dict.values():
        if typ in prog_typs:
            prog_count += 1
    type_progcount[typ] = prog_count

print(sorted(type_progcount.items(), key=itemgetter(1))[::-1])
'''

## print most common 100 types in the dataset
#print(Counter([x['output'] for x in dataset]).most_common(100))


## below is useful for visualizing the distribution of type counts in the dataset
'''
counter = Counter([x['output'] for x in dataset])
type_counts = [counter[x] for x in counter]
plt.hist(type_counts, bins=[10, 50, 100, 300, 500 ])#1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000])
plt.title("Distribution of Types")
plt.show()
'''

def choose_top_labels(dataset, prog_type_dict, label_choice, label_num, min_prognum_labels):
    if (label_choice == "TOP"):
        return [x[0] for x in Counter([x['output'] for x in dataset]).most_common(label_num)]
    elif (label_choice == "PROG"):
        ## first count how many programs each type appears in
        type_progcount = {}
        for typ in Counter([x['output'] for x in dataset]).keys():
            prog_count = 0
            for prog_typs in prog_type_dict.values():
                if typ in prog_typs:
                    prog_count += 1
            type_progcount[typ] = prog_count
        return [key for key,value in type_progcount.items() if value >= min_prognum_labels]
    else:
        print('unexpected value of label_choice: {}'.format(label_choice))
        raise ValueError

def create_labels(dataset, prog_type_dict, label_choice, label_num=100, min_prognum_labels=5):
    ## create label indices, and dict to map back from index to label name
    top_labels = choose_top_labels(dataset, prog_type_dict, label_choice, label_num, min_prognum_labels)
    ## plus 1 needed below because we use these as conditional guards later on
    label_to_idx = {x: top_labels.index(x) for x in top_labels}
    idx_to_label = {v: k for k,v in label_to_idx.items()}

    ## print test below
    #print("string to idx: {}".format(label_to_idx['hash']))
    #print("idx to string: {}".format(idx_to_label[3]))
    return [label_to_idx, idx_to_label]

    
def create_tokenizer(dataset):
    ## create char tokenizer / "fit" to the characters in the inputs
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True)
    lang_tokenizer.fit_on_texts([x['input'] for x in dataset])
    return lang_tokenizer



## test out tokenizer
'''
in_string = dataset[0]['input']
token_string = lang_tokenizer.texts_to_sequences(in_string)
tok_to_string = lang_tokenizer.sequences_to_texts(token_string)

## MULTIPLE INPUTS
#tok_to_string = lang_tokenizer.texts_to_sequences([x['input'] for x in dataset[:5]]) 

print("Input: {}".format(in_string))
print("Tokenized: {}".format(token_string))
print("Output from Tokens: {}".format(tok_to_string))
'''

def split_train_dev(dataset):
    train_size = int(0.9 * len(dataset))
    dev_size = len(dataset) - train_size
    train_dataset, dev_dataset = torch.utils.data.random_split(dataset, [train_size, dev_size])
    return [train_dataset, dev_dataset]

def prepare_data(dataset, lang_tokenizer, label_to_idx):
    ## prepare input/output data
    input_data = []
    output_data = []

    for sample in dataset:
        ## only use data which falls in top N types
        ## TODO: do we want to have an "other" type?
        if label_to_idx.get(sample['output'], -1) > -1:
            input_data.append(lang_tokenizer.texts_to_sequences(sample['input']))
            output_data.append(label_to_idx[sample['output']])


    ## pad sequences so they're all same length
    in_data = tf.keras.preprocessing.sequence.pad_sequences(input_data).squeeze()

    ## slice in/out data together to create dataset
    return tf.data.Dataset.from_tensor_slices((in_data, output_data))


'''
dataset, prog_type_dict = create_names_dataset(DATA_FILE)
label_to_idx, idx_to_label = create_labels(dataset, prog_type_dict)
num_labels = len(label_to_idx)
lang_tokenizer = create_tokenizer(dataset)
vocab_size = max(lang_tokenizer.index_word.keys())
## SAVE TOKENIZER
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(lang_tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
train_dataset, dev_dataset = split_train_dev(dataset)
train_ds = prepare_data(train_dataset, label_to_idx)
dev_ds = prepare_data(dev_dataset, label_to_idx)
# shuffle and create batches
batch_size = 128
train_ds = train_ds.shuffle(20000).batch(batch_size)
dev_ds = dev_ds.shuffle(20000).batch(batch_size)

## initialize/compile model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size + 1, 128, mask_zero=True),
    #tf.keras.layers.Masking(mask_value=0.,input_shape=(66, vocab_size)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_labels)
])

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])


## TRAINING
## uncomment below to train/save model
history = model.fit(train_ds, epochs=10, validation_data=dev_ds)
model.save("id_char_model.h5")

## LOAD SAVED MODEL
#model = load_model("id_char_model.h5")

## test out the model on ID name "str"
#x = model.predict(lang_tokenizer.texts_to_sequences(["str"]))

## convert max output back into label, print
#print(idx_to_label[np.argmax(x)])
'''




