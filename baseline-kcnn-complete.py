import json
import csv
import numpy as np
import pandas as pd
import tqdm
import time
import sys
import torch

from datetime import datetime

from tensorflow.python.client import device_lib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from keras.models import Model
from keras.layers import *
from keras.utils.np_utils import to_categorical
from keras import regularizers

import re

import work

ENV_PATH = f'./dataset/{sys.argv[1]}'
TARGET_LABEL = sys.argv[2]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    works = work.read_cleaned_works(f"{ENV_PATH}/cleaned")
    work_list = work.get_work_list_by_label(works, TARGET_LABEL)
    works_df = pd.DataFrame(work_list)
    true_df = works_df[works_df['label'] == 1]
    false_df = works_df[works_df['label'] == 0].sample(n=len(true_df))
    data_df = pd.concat([true_df, false_df]).sample(frac=1)
    
    max_features = 65535
    sequence_length = 16384
    
    tokenizer = Tokenizer(num_words=max_features, split=' ', char_level='True', oov_token='<unw>')
    tokenizer.fit_on_texts(data_df['content'].values)
    
    x = tokenizer.texts_to_sequences(data_df['content'].values)
    x = pad_sequences(x, sequence_length)
    y = data_df['label'].values
    
    total_count = len(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
    print(f'Train/Valid: {len(x_train)}, {len(y_train)}')
    print(f'Test: {len(x_test)}, {len(y_test)}')
    
    #x_train = torch.from_numpy(x_train)
    #x_train = x_train.to(device)
    
    embedding_dim = 300 # Kim uses 300 here
    num_filters = 100
    num_words = 16384
    batch_size = 32
    
    inputs = Input(shape=(sequence_length,), dtype='int32')
    embedding_layer = Embedding(input_dim=max_features, output_dim=embedding_dim, input_length=sequence_length)(inputs)

    reshape = Reshape((sequence_length, embedding_dim, 1))(embedding_layer)

    conv_0 = Conv2D(num_filters, kernel_size=(3, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape)
    conv_1 = Conv2D(num_filters, kernel_size=(4, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape)
    conv_2 = Conv2D(num_filters, kernel_size=(5, embedding_dim), padding='valid', kernel_initializer='normal', activation='relu', kernel_regularizer=regularizers.l2(3))(reshape)

    maxpool_0 = MaxPool2D(pool_size=(sequence_length - 3 + 1, 1), strides=(1,1), padding='valid')(conv_0)
    maxpool_1 = MaxPool2D(pool_size=(sequence_length - 4 + 1, 1), strides=(1,1), padding='valid')(conv_1)
    maxpool_2 = MaxPool2D(pool_size=(sequence_length - 5 + 1, 1), strides=(1,1), padding='valid')(conv_2)

    concatenated_tensor = Concatenate(axis=1)([maxpool_0, maxpool_1, maxpool_2])
    flatten = Flatten()(concatenated_tensor)

    dropout = Dropout(0.5)(flatten)
    # note the different activation
    output = Dense(units=1, activation='sigmoid')(dropout)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    st_time = time.time()
    ### EPOCH -> 3 FOR TESTING ###
    history = model.fit(x_train, y_train, epochs=30, batch_size=batch_size, verbose=1,validation_split=0.1, shuffle=True)
    
    val_acc = history.history['val_accuracy'][-1]
    
    str_date = datetime.today().strftime('%Y-%m-%d')
    model.save(f'{ENV_PATH}/results/KimCNN/{TARGET_LABEL}/{str_date}')
    t_time = time.strftime("%H:%M:%S", time.gmtime(time.time()-st_time))
    
    y_pred = model.predict(x_test)
    #print(y_pred)
    #print(y_test)
    for i, y in enumerate(y_pred):
        if y[0] > 0.49:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    report = classification_report(y_test, y_pred, target_names=['False', 'True'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.applymap('{:.4f}'.format).to_csv(f'{ENV_PATH}/results/KimCNN/{TARGET_LABEL}/{str_date}.csv')
    acc = report['accuracy']
    #print(report)
    
    with open(f'{ENV_PATH}/results/KimCNN/{str_date}.jsonl', 'a+', encoding='utf-8') as file:
        now = datetime.now()
        str_time = now.strftime("%Y-%m-%d %H:%M:%S")
        data = {"label": TARGET_LABEL, "time": str_time, "training time": t_time, "total": total_count, "accuracy": float("{:.4f}".format(acc)), "valid_acc": float("{:.4f}".format(val_acc))}
        json.dump(data, file)
        file.write("\n")    
    
if __name__ == "__main__":
    print(f'Start retraining: Target = {TARGET_LABEL}')
    main()
    print("\ndone!\n")