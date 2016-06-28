#pylint: skip-file
import time
import sys
import numpy as np
import theano
import theano.tensor as T
#from utils_pg import *
from VAE import *
import data

use_gpu(0)

e = 0.01
lr = 1
drop_rate = 0.
batch_size = 128
hidden_size = 400
latent_size = 50
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "adadelta"
continuous = False

if continuous:
    pass
else:
    train_xy, valid_xy, test_xy = data.mnist(batch_size)

dim_x = train_xy[0][0].shape[1]
dim_y = train_xy[0][1].shape[1]
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = VAE(dim_x, dim_x, hidden_size, latent_size, optimizer)

print "training..."
start = time.time()
for i in xrange(50):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in train_xy.items():
        X = xy[0] 
        cost = model.train(X, lr)
        error += cost
    in_time = time.time() - in_start

    error /= len(train_xy);
    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)
    if error <= e:
        break

print "training finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/vae_mnist.model", model)

print "validation.."
load_model("./model/vae_mnist.model", model)
error = 0
for batch_id, xy in valid_xy.items():
    X = xy[0]
    cost, y = model.predict(X)
    error += cost
print "Loss = " + str(error / len(valid_xy))
