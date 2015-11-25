from __future__ import absolute_import
import numpy as np

import util
from os import path
import itertools
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import KFold
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D

'''
    Train a recurrent convolutional network on the IMDB sentiment classification task.
    GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_lstm.py
    Get to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
nb_epoch = 2
n_fold = 5


def cnn_lstm_model(X_train, y_train, X_test, y_test):
    print("Pad sequences (samples x time)")
    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)

    print('Build model...')

    model = Sequential()
    model.add(Embedding(max_features, embedding_size, input_length=maxlen))
    model.add(Dropout(0.25))
    model.add(Convolution1D(nb_filter=nb_filter,
                            filter_length=filter_length,
                            border_mode="valid",
                            activation="relu",
                            subsample_length=1))
    model.add(MaxPooling1D(pool_length=pool_length))
    model.add(LSTM(lstm_output_size))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  class_mode="binary")

    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
                                show_accuracy=True)
    pred_labels = model.predict_classes(X_test)
    # print pred_labels
    precision, recall, f1, supp = precision_recall_fscore_support(y_test, pred_labels, average='weighted')
    print precision, recall, f1, supp

    return precision, recall, f1


def build_dataset(user_data):
    tokenizer = Tokenizer(nb_words=1000)
    all_review_user = ""
    for single_example in user_data:
        all_review_user += single_example['rev'].encode('utf-8')
    tokenizer.fit_on_texts(all_review_user)
    X = []
    y = []
    for single_example in user_data:
        rating = int(float(single_example['rat']))

        review_seq = tokenizer.texts_to_sequences(single_example['rev'].encode('utf-8'))
        # print review_seq
        x = list(itertools.chain(*review_seq))
        X.append(x)
        y.append(rating)
        # break
    # X = sequence.pad_sequences(X, maxlen=max_len)
    X = np.asarray(X)
    y = np.asarray(y)

    return X, y


def execute_model(X, y):
    kf = KFold(y.shape[0], n_folds=n_fold, shuffle=True)
    results_user = np.array([0.0, 0.0, 0.0])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        precision, recall, f1 = cnn_lstm_model(X_train, y_train, X_test, y_test)
        # precision, recall, f1 = bidirectional_lstm(X_train, y_train, X_test, y_test)
        results_user[0] += precision
        results_user[1] += recall
        results_user[2] += f1
    results_user /= n_fold
    return results_user

if __name__ == '__main__':
    n_count = 0
    data = util.read_data(path.join(util.data_path, util.file_name))
    X, y = build_dataset(data)
    results = execute_model(X, y)
    print results