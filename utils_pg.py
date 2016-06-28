#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
import cPickle as pickle

# set use gpu programatically
import theano.sandbox.cuda
def use_gpu(gpu_id):
    if gpu_id > -1:
        theano.sandbox.cuda.use("gpu" + str(gpu_id))

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape, name, sample = "xavier"):
    if sample == "uniform":
        values = np.random.uniform(-0.08, 0.08, shape)
    elif sample == "xavier":
        values = np.random.uniform(-np.sqrt(6. / (shape[0] + shape[1])), np.sqrt(6. / (shape[0] + shape[1])), shape)
    elif sample == "ortho":
        W = np.random.randn(shape[0], shape[0])
        u, s, v = np.linalg.svd(W)
        values = u
    else:
        raise ValueError("Unsupported initialization scheme: %s" % sample)
    
    return theano.shared(floatX(values), name)

def init_gradws(shape, name):
    return theano.shared(floatX(np.zeros(shape)), name)

def init_bias(size, name):
    return theano.shared(floatX(np.zeros((size,))), name)

def init_mat(mat, name):
    return theano.shared(floatX(mat), name)

def save_model(f, model):
    ps = {}
    for p in model.params:
        ps[p.name] = p.get_value()
    pickle.dump(ps, open(f, "wb"))

def load_model(f, model):
    ps = pickle.load(open(f, "rb"))
    for p in model.params:
        p.set_value(ps[p.name])
    return model
