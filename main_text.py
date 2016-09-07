#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
from VAE import *
import data
import matplotlib.pyplot as plt

use_gpu(0)

lr = 0.001
drop_rate = 0.
batch_size = 20
hidden_size = 500
latent_size = 2
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "adam"
continuous = False

train_idx, valid_idx, test_idx, other_data = data.apnews()
[docs, dic, w2i, i2w] = other_data

dim_x = len(dic)
dim_y = dim_x
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = VAE(dim_x, dim_x, hidden_size, latent_size, continuous, optimizer)

print "training..."
start = time.time()
for i in xrange(10):
    train_xy = data.batched_idx(train_idx, batch_size)
    error = 0.0
    in_start = time.time()
    for batch_id, x_idx in train_xy.items():
        X = data.batched_news(x_idx, other_data)
        cost, z = model.train(X, lr)
        error += cost
        #print i, batch_id, "/", len(train_xy), cost
    in_time = time.time() - in_start

    error /= len(train_xy);
    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)

print "training finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/vae_text.model", model)

print "lode model..."
load_model("./model/vae_text.model", model)

print "validation.."
valid_xy = data.batched_idx(valid_idx, batch_size)
error = 0
for batch_id, x_idx in valid_xy.items():
    X = data.batched_news(x_idx, other_data)
    cost, y = model.validate(X)
    error += cost
print "Loss = " + str(error / len(valid_xy))

top_w = 20
## manifold 
if latent_size == 2:
    test_xy = data.batched_idx(test_idx, 1000)
    x_idx = test_xy[0]
    X = data.batched_news(x_idx, other_data)

    mu = np.array(model.project(X))
    
    plt.figure(figsize=(8, 6)) 
    plt.scatter(mu[:, 0], mu[:, 1], c="r")
    #plt.savefig("2dstructure.png", bbox_inches="tight")
    plt.show()

    nx = ny = 20
    v = 100
    x_values = np.linspace(-v, v, nx)
    y_values = np.linspace(-v, v, ny) 
    canvas = np.empty((28*ny, 20*nx))
    for i, xi in enumerate(x_values):
        for j, yi in enumerate(y_values):
            z = np.array([[xi, yi]], dtype=theano.config.floatX)
            y = model.generate(z)[0,:]
            ind = np.argsort(-y)
            print xi, yi, 
            for k in xrange(top_w):
                print i2w[ind[k]],
            print "\n"

