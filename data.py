# -*- coding: utf-8 -*-
#pylint: skip-file
import sys
import os
import numpy as np
import theano
import theano.tensor as T
import cPickle, gzip
import string

curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))

#data: http://deeplearning.net/data/mnist/mnist.pkl.gz
def mnist():
    f = gzip.open(curr_path + "/data/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    return train_set, valid_set, test_set

def freyface():
    raw_faces = cPickle.load(open(curr_path + "/data/freyfaces.pkl", "rb"))
    mat_faces = np.zeros((len(raw_faces), len(raw_faces[0])))
    for i in range(len(raw_faces)): # 1965 in total
        mat_faces[i, :] = np.asarray(raw_faces[i])

    train_set = mat_faces[:1600, :]
    valid_set = mat_faces[1600:1800, :]
    test_set = mat_faces[1800:, :]
    return (train_set, ), (valid_set, ), (test_set, )

def batched_mnist(data_set, batch_size = 1):
    lst = [n for n in range(len(data_set[0]))]
    np.random.shuffle(lst)
    X = data_set[0][lst,]
    Y = data_set[1][lst]

    data_xy = {}
    batch_x = []
    batch_y = []
    batch_id = 0
    for i in xrange(len(X)):
        batch_x.append(X[i])
        y = np.zeros((10), dtype = theano.config.floatX)
        y[Y[i]] = 1
        batch_y.append(y)
        if (len(batch_x) == batch_size) or (i == len(X) - 1):
            data_xy[batch_id] = [np.matrix(batch_x, dtype = theano.config.floatX), \
                                     np.matrix(batch_y, dtype = theano.config.floatX)]
            batch_id += 1
            batch_x = []
            batch_y = []
    return data_xy

def batched_freyface(data_set, batch_size = 1):
    lst = [n for n in range(len(data_set[0]))]
    np.random.shuffle(lst)
    data_xy = {}
    batch_x = []
    X = data_set[0][lst,]
    batch_id = 0
    for i in xrange(len(X)):
        batch_x.append(X[i])
        if (len(batch_x) == batch_size) or (i == len(X) - 1):
            data_xy[batch_id] = [np.matrix(batch_x, dtype = theano.config.floatX)]
            batch_id += 1
            batch_x = []
    return data_xy

#data: http://deeplearning.net/data/mnist/mnist.pkl.gz
def shared_mnist():
    def shared_dataset(data_xy):
        data_x, data_y = data_xy
        np_y = np.zeros((len(data_y), 10), dtype=theano.config.floatX)
        for i in xrange(len(data_y)):
            np_y[i, data_y[i]] = 1

        shared_x = theano.shared(np.asmatrix(data_x, dtype=theano.config.floatX))
        shared_y = theano.shared(np.asmatrix(np_y, dtype=theano.config.floatX))
        return shared_x, shared_y
    f = gzip.open(curr_path + "/data/mnist.pkl.gz", "rb")
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    return [train_set_x, train_set_y], [valid_set_x, valid_set_y], [test_set_x, test_set_y]

def load_stopwords():
    stop_words = {}
    f = open("./data/stopwords.txt", "r")
    for line in f:
        line = line.strip('\n').strip()
        stop_words[line] = 1
    return stop_words

def apnews():
    dic = {}
    i2w = {}
    w2i = {}
    docs = {}
    stop_words = load_stopwords()
    
    f = open("./data/news_ap.txt", "r")
    doc_id = 0
    for line in f:
        line = line.strip('\n').lower()
        line = line.translate(None, string.punctuation)
        words = line.split()
        d = []
        for w in words:
            if w in stop_words:
                continue
            d.append(w)
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
                w2i[w] = len(i2w)
                i2w[len(i2w)] = w

        docs[doc_id] = d
        doc_id += 1
    f.close()

    print len(docs), len(w2i), len(i2w), len(dic)
    doc_idx = [i for i in xrange(len(docs))]
    spliter = (int) (len(docs) / 10.0 * 9)
    train_idx = doc_idx[0:spliter]
    valid_idx = doc_idx[spliter:len(docs)]
    test_idx = valid_idx

    return train_idx, valid_idx, test_idx, [docs, dic, w2i, i2w]

def batched_idx(lst, batch_size = 1):
    np.random.shuffle(lst)
    data_xy = {}
    batch_x = []
    batch_id = 0
    for i in xrange(len(lst)):
        batch_x.append(lst[i])
        if (len(batch_x) == batch_size) or (i == len(lst) - 1):
            data_xy[batch_id] = batch_x
            batch_id += 1
            batch_x = []
    return data_xy

def batched_news(x_idx, data):
    [docs, dic, w2i, i2w] = data
    X = np.zeros((len(x_idx), len(dic)), dtype = theano.config.floatX)    
    for i in xrange(len(x_idx)):
        xi = x_idx[i]
        d = docs[xi]
        for w in d:
            X[i, w2i[w]] += 1
   
    for i in xrange(len(x_idx)):
        norm2 = np.linalg.norm(X[i,:])
        if norm2 != 0:
            X[i,:] /= norm2
    
    return X



