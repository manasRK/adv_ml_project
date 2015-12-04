# gensim modules
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec

# numpy
import numpy as np
import json
from os import path
import util
# classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support,accuracy_score
from sklearn.cross_validation import KFold

n_fold = 5

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        return np.random.permutation(self.sentences)


def write_data(full_file_name):
    total_pos = 0
    total_neg = 0
    raw_file = open(full_file_name, 'r')
    data = json.load(raw_file)
    w1 = open('pos.txt', 'wb+')
    w2 = open('neg.txt', 'wb+')
    for entry in data:
        rat = entry['rat']
        if float(rat) == 1.0:
            w1.write(entry['rev'].encode('utf-8') + '\n')
            total_pos += 1
        else:
            w2.write(entry['rev'].encode('utf-8') + '\n')
            total_neg += 1
    return total_pos, total_neg


def execute_model(X, y):
    print X.shape, y.shape
    kf = KFold(y.shape[0], n_folds=n_fold, shuffle=True)
    results_user = np.array([0.0, 0.0, 0.0, 0.0])
    for train_index, test_index in kf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        accuracy, precision, recall, f1 = prediction_model(X_train, y_train, X_test, y_test)
        results_user[0] += accuracy
        results_user[1] += precision
        results_user[2] += recall
        results_user[3] += f1
    results_user /= n_fold
    return results_user


def prediction_model(X_train, y_train, X_test, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    pred_labels = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, pred_labels)
    precision, recall, f1, supp = precision_recall_fscore_support(y_test, pred_labels, average='weighted')
    return accuracy, precision, recall, f1


if __name__ == '__main__':
    total_pos, total_neg = write_data(path.join(util.data_path, util.file_name))
    # print total_neg, total_pos
    sources = {'pos.txt': 'POS', 'neg.txt': 'NEG'}
    sentences = LabeledLineSentence(sources)
    model = Doc2Vec(size=100, alpha=0.025, min_alpha=0.025, window=5)
    model.build_vocab(sentences.to_array())
    for epoch in xrange(0, 10):
        model.train(sentences)
        print 'end of iteration ', epoch + 1
    X = []
    y = []

    for i in xrange(0, total_pos):
        prefix_pos = 'POS' + '_%s' % i
        X.append(model.docvecs[prefix_pos])
        y.append(1)
    for i in xrange(0, total_neg):
        prefix_neg = 'NEG' + '_%s' % i
        X.append(model.docvecs[prefix_neg])
        y.append(0)
    X = np.asarray(X)
    y = np.asarray(y)
    results = execute_model(X, y)
    print results
    util.insert_results('Doc2Vec', results[0], results[2], results[1], results[3])
