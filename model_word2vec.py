import json
from gensim import utils
from os import path
import util
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.cross_validation import KFold

n_fold = 5


class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        raw_file = open(self.sources, 'r')
        data = json.load(raw_file)
        for entry_no, entry in enumerate(data):
            label = entry['uid'] + '_' + entry['pid']
            yield LabeledSentence(entry['rev'].encode('utf-8').split(), [label])


def get_labels_ratings(full_file_name):
    raw_file = open(full_file_name, 'r')
    data = json.load(raw_file)
    labels_ratings = []
    for entry in data:
        label = entry['uid'] + '_' + entry['pid']
        rating = entry['rat']
        labels_ratings.append((label.encode('utf-8'), rating))
    return labels_ratings


def read_labeled_doc(full_file_name):
    raw_file = open(full_file_name, 'r')
    data = json.load(raw_file)
    for entry in data:
        label = entry['uid'] + '_' + entry['pid']
        review = entry['rev']
        rat = entry['rat']
        yield LabeledSentence(words=review.split(), tags=[label.encode('utf-8')]), rat


def execute_model(X, y):
    print X.shape, y.shape
    kf = KFold(y.shape[0], n_folds=n_fold, shuffle=True)
    results_user = np.array([0.0, 0.0, 0.0])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        precision, recall, f1 = prediction_model(X_train, y_train, X_test, y_test)
        results_user[0] += precision
        results_user[1] += recall
        results_user[2] += f1
    results_user /= n_fold
    return results_user


def prediction_model(X_train, y_train, X_test, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    pred_labels = classifier.predict(X_test)
    precision, recall, f1 = precision_recall_fscore_support(y_test, pred_labels, average='weighted')
    return precision, recall, f1

if __name__ == '__main__':
    model = Doc2Vec(alpha=0.025, min_alpha=0.025, size=100, window=10, negative=5)
    model.build_vocab(read_labeled_doc(path.join(util.data_path, util.file_name))[0])
    for i in xrange(0, 1):
        model.train(read_labeled_doc(path.join(util.data_path, util.file_name)))
        print 'end of iteration ', i + 1
    labels_ratings = get_labels_ratings(path.join(util.data_path, util.file_name))
    X = []
    y = []

    for entry in labels_ratings:
        current_data = model.docvecs[entry[0]]
        current_rating = labels_ratings[entry[1]]
        X.append(current_data)
        y.append(y)
    X = np.asarray(X)
    y = np.asarray(y)
    results = execute_model(X, y)
    print results

