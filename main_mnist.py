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

train_set, valid_set, test_set = data.mnist()

train_xy = data.batched_mnist(train_set, batch_size)
dim_x = train_xy[0][0].shape[1]
dim_y = train_xy[0][1].shape[1]
print "#features = ", dim_x, "#labels = ", dim_y

print "compiling..."
model = VAE(dim_x, dim_x, hidden_size, latent_size, continuous, optimizer)

print "training..."
start = time.time()
for i in xrange(50):
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
save_model("./model/vae_mnist.model", model)


'''-------------Visualization------------------'''
# code from: https://jmetzen.github.io/2015-11-27/vae.html

load_model("./model/vae_mnist.model", model)

print "validation.."
valid_xy = data.batched_mnist(valid_set, batch_size)
error = 0
for batch_id, xy in valid_xy.items():
    X = xy[0]
    cost, y = model.validate(X)
    error += cost
print "Loss = " + str(error / len(valid_xy))

plt.figure(figsize=(8, 12))
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
plt.savefig("reconstruct.png", bbox_inches="tight")
plt.show()

## manifold 
if latent_size == 2:
    test_xy = data.batched_mnist(test_set, 5000)
    X = test_xy[0][0]

    mu = np.array(model.project(X))
    
    plt.figure(figsize=(8, 6)) 
    plt.scatter(mu[:, 0], mu[:, 1], c=np.argmax(np.array(test_xy[0][1]), 1))
    plt.colorbar()
    plt.savefig("2dstructure.png", bbox_inches="tight")
    plt.show()
    
    '''--------------------------'''
    nx = ny = 20
    x_values = np.linspace(-3, 3, nx)
    y_values = np.linspace(-3, 3, ny) 
    canvas = np.empty((28*ny, 28*nx))
    for i, yi in enumerate(x_values):
        for j, xi in enumerate(y_values):
            z = np.array([[xi, yi]], dtype=theano.config.floatX)
            y = model.generate(z)
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = y.reshape(28, 28)

    fit = plt.figure(figsize=(8, 10))        
    Xi, Yi = np.meshgrid(x_values, y_values)
    plt.imshow(canvas, origin="upper")
    plt.tight_layout()
    plt.savefig("manifold.png", bbox_inches="tight")
    plt.show()

