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
batch_size = 128
hidden_size = 500
latent_size = 2
# try: sgd, momentum, rmsprop, adagrad, adadelta, adam, nesterov_momentum
optimizer = "adam"
continuous = False

train_set, valid_set, test_set = data.freyface()

train_xy = data.batched_freyface(train_set, batch_size)
dim_x = train_xy[0][0].shape[1]
dim_y = dim_x
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = VAE(dim_x, dim_x, hidden_size, latent_size, continuous, optimizer)

print "training..."
start = time.time()
for i in xrange(100):
    error = 0.0
    in_start = time.time()
    for batch_id, xy in train_xy.items():
        X = xy[0] 
        cost, z = model.train(X, lr)
        error += cost
    in_time = time.time() - in_start

    error /= len(train_xy);
    print "Iter = " + str(i) + ", Loss = " + str(error) + ", Time = " + str(in_time)

print "training finished. Time = " + str(time.time() - start)

print "save model..."
save_model("./model/vae_face.model", model)

'''-------------Visualization------------------'''
# code from: https://jmetzen.github.io/2015-11-27/vae.html

load_model("./model/vae_face.model", model)

print "validation.."
valid_xy = data.batched_freyface(valid_set, batch_size)
error = 0
for batch_id, xy in valid_xy.items():
    X = xy[0]
    cost, y = model.validate(X)
    error += cost
print "Loss = " + str(error / len(valid_xy))

plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(X[i].reshape(28, 20), vmin=0, vmax=1, cmap='gray_r')
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(y[i].reshape(28, 20), vmin=0, vmax=1, cmap='gray_r')
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
plt.savefig("reconstruct.png", bbox_inches="tight")
plt.show()

## manifold 
if latent_size == 2:
    test_xy = data.batched_freyface(test_set, 160)
    X = test_xy[0][0]

    mu = np.array(model.project(X))
    
    plt.figure(figsize=(8, 6)) 
    plt.scatter(mu[:, 0], mu[:, 1], c="r")
    plt.savefig("2dstructure.png", bbox_inches="tight")
    plt.show()
    
    #################

    nx = ny = 20
    v = 3
    x_values = np.linspace(-v, v, nx)
    y_values = np.linspace(-v, v, ny) 
    canvas = np.empty((28*ny, 20*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z = np.array([[xi, yi]], dtype=theano.config.floatX)
            y = model.generate(z)
            canvas[(nx-i-1)*28:(nx-i)*28, j*20:(j+1)*20] = y.reshape(28, 20)

    fit = plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper", cmap='gray_r')
    plt.tight_layout()
    plt.savefig("manifold.png", bbox_inches="tight")
    plt.show()

