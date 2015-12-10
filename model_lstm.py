from os import path
import itertools
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.cross_validation import KFold
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from seya.layers.recurrent import Bidirectional
import util

'''GPU command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python model_lstm.py
'''

test_split = 0.2
max_len = 100
max_features = 20000
batch_size = 32
n_fold = 5


def bidirectional_lstm(X_train, y_train, X_test, y_test):
    X_train = sequence.pad_sequences(X_train, maxlen=max_len)
    X_test = sequence.pad_sequences(X_test, maxlen=max_len)
    lstm = LSTM(output_dim=64)
    gru = GRU(output_dim=64)  # original examples was 128, we divide by 2 because results will be concatenated
    brnn = Bidirectional(forward=lstm, backward=gru)
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_len))
    model.add(brnn)  # try using another Bidirectional RNN inside the Bidirectional RNN. Inception meets callback hell.
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")
    # model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
    score, acc = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
    pred_labels = model.predict_classes(X_test)
    # print pred_labels
    accuracy = accuracy_score(y_test, pred_labels)
    precision, recall, f1, supp = precision_recall_fscore_support(y_test, pred_labels, average='weighted')
    print precision, recall, f1, supp

    return accuracy, precision, recall, f1


def lstm_model(X_train, y_train, X_test, y_test):
    X_train = sequence.pad_sequences(X_train, maxlen=max_len, padding='post')
    X_test = sequence.pad_sequences(X_test, maxlen=max_len, padding='post')
    print X_train.shape, y_train.shape
    print X_test.shape, y_test.shape

    print('Build model...')
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_len))
    model.add(LSTM(128))  # try using a GRU instead, for fun
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    # print X_train.shape, y_train.shape
    # print X_test.shape, y_test.shape

    model.compile(loss='binary_crossentropy', optimizer='adam', class_mode="binary")

    print("Train...")
    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)
    acc, score = model.evaluate(X_test, y_test, batch_size=batch_size, show_accuracy=True)
    print('Test score:', score)
    print('Test accuracy:', acc)
    pred_labels = model.predict_classes(X_test)
    # print pred_labels
    accuracy = accuracy_score(y_test, pred_labels)
    precision, recall, f1, supp = precision_recall_fscore_support(y_test, pred_labels, average='weighted')
    print precision, recall, f1, supp

    return accuracy, precision, recall, f1


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
    results_user = np.array([0.0, 0.0, 0.0, 0.0])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # accuracy, precision, recall, f1 = lstm_model(X_train, y_train, X_test, y_test)
        accuracy, precision, recall, f1 = bidirectional_lstm(X_train, y_train, X_test, y_test)
        results_user[0] += accuracy
        results_user[1] += precision
        results_user[2] += recall
        results_user[3] += f1
    results_user /= n_fold
    return results_user


if __name__ == '__main__':
    n_count = 0
    data = util.read_data(path.join(util.data_path, util.file_name))
    X, y = build_dataset(data)
    results = execute_model(X, y)
    print results
    util.insert_results('BI-LSTM', results[0], results[2], results[1], results[3])
