import util
from os import path
import itertools
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils

from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import KFold
'''GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python model_lstm.py
'''

test_split = 0.2
max_len = 1000
max_features = 2000
batch_size = 32
nb_epoch = 1
n_fold = 5


def build_dataset(data):
    tokenizer = Tokenizer(nb_words=1000)
    all_review_user = ""
    for single_example in data:
        all_review_user += single_example['rev'].encode('utf-8')
    tokenizer.fit_on_texts(all_review_user)
    X = []
    y = []
    for single_example in data:
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
    print X.shape, y.shape
    kf = KFold(y.shape[0], n_folds=n_fold, shuffle=True)
    results_user = np.array([0.0, 0.0, 0.0])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        precision, recall, f1 = mlp_model(X_train, y_train, X_test, y_test)
        results_user[0] += precision
        results_user[1] += recall
        results_user[2] += f1
    results_user /= n_fold
    return results_user


def mlp_model(X_train, y_train, X_test, y_test):
    tokenizer = Tokenizer(nb_words=1000)
    nb_classes = np.max(y_train) + 1

    X_train = tokenizer.sequences_to_matrix(X_train, mode="freq")
    X_test = tokenizer.sequences_to_matrix(X_test, mode="freq")

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print("Building model...")
    model = Sequential()
    model.add(Dense(512, input_shape=(max_len,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode='categorical')

    history = model.fit(X_train, Y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, show_accuracy=True, validation_split=0.1)
    model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=1, show_accuracy=True)
    # print('Test score:', score[0])
    # print('Test accuracy:', score[1])
    pred_labels = model.predict_classes(X_test)
    # print pred_labels
    # print y_test
    precision, recall, f1, supp = precision_recall_fscore_support(y_test, pred_labels, average='weighted')
    print precision, recall, f1, supp

    return precision, recall, f1

if __name__ == '__main__':
    n_count = 0
    data = util.read_data(path.join(util.data_path, util.file_name))
    X, y = build_dataset(data)
    results = execute_model(X, y)
    print results