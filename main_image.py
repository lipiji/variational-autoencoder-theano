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

e = 0.01
lr = 0.001
drop_rate = 0.
batch_size = 128
hidden_size = 500
latent_size = 2
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "adam"
continuous = False

if continuous:
    pass
else:
    train_set, valid_set, test_set = data.mnist()

train_xy = data.batched(train_set, batch_size)
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

load_model("./model/vae_mnist.model", model)

print "validation.."
valid_xy = data.batched(valid_set, batch_size)
error = 0
for batch_id, xy in valid_xy.items():
    X = xy[0]
    cost, y = model.valid(X)
    error += cost
print "Loss = " + str(error / len(valid_xy))


fig = plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(X[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(y[i].reshape(28, 28), vmin=0, vmax=1)
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.show()
fig.savefig("reconstruct.png", bbox_inches="tight")
plt.close(fig)

if latent_size == 2:
    test_xy = data.batched(test_set, 5000)
    X = test_xy[0][0]
    mu = np.array(model.project(X))
    
    fig = plt.figure(figsize=(8, 6)) 
    plt.scatter(mu[:, 0], mu[:, 1], c=np.argmax(np.array(test_xy[0][1]), 1))
    plt.colorbar()
    plt.show()
    fig.savefig("2dstructure.png", bbox_inches="tight")
    plt.close(fig)
