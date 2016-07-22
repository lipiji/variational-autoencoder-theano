#pylint: skip-file
import numpy as np
import theano
import theano.tensor as T
from utils_pg import *
from updates import *

class VAE(object):
    def __init__(self, in_size, out_size, hidden_size, latent_size, continuous, optimizer = "adadelta"):
        self.prefix = "VAE_"
        self.X = T.matrix("X")
        self.in_size = in_size
        self.out_size = out_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.optimizer = optimizer
        self.continuous = continuous

        self.define_layers()
        self.define_train_test_funcs()
        
    def define_layers(self):
        self.params = []
        
        layer_id = "1"
        self.W_xh = init_weights((self.in_size, self.hidden_size), self.prefix + "W_xh" + layer_id)
        self.b_xh = init_bias(self.hidden_size, self.prefix + "b_xh" + layer_id)

        layer_id = "2"
        self.W_hu = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hu" + layer_id)
        self.b_hu = init_bias(self.latent_size, self.prefix + "b_hu" + layer_id)
        self.W_hsigma = init_weights((self.hidden_size, self.latent_size), self.prefix + "W_hsigma" + layer_id)
        self.b_hsigma = init_bias(self.latent_size, self.prefix + "b_hsigma" + layer_id)

        layer_id = "3"
        self.W_zh = init_weights((self.latent_size, self.hidden_size), self.prefix + "W_zh" + layer_id)
        self.b_zh = init_bias(self.hidden_size, self.prefix + "b_zh" + layer_id)
 
        self.params += [self.W_xh, self.b_xh, self.W_hu, self.b_hu, self.W_hsigma, self.b_hsigma, \
                        self.W_zh, self.b_zh]

        layer_id = "4"
        if self.continuous:
            self.W_hyu = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hyu" + layer_id)
            self.b_hyu = init_bias(self.out_size, self.prefix + "b_hyu" + layer_id)
            self.W_hysigma = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hysigma" + layer_id)
            self.b_hysigma = init_bias(self.out_size, self.prefix + "b_hysigma" + layer_id)
            self.params += [self.W_hyu, self.b_hyu, self.W_hysigma, self.b_hysigma]
        else:
            self.W_hy = init_weights((self.hidden_size, self.out_size), self.prefix + "W_hy" + layer_id)
            self.b_hy = init_bias(self.out_size, self.prefix + "b_hy" + layer_id)
            self.params += [self.W_hy, self.b_hy]

        # encoder
        h_enc = T.nnet.relu(T.dot(self.X, self.W_xh) + self.b_xh)
        
        self.mu = T.dot(h_enc, self.W_hu) + self.b_hu
        log_var = T.dot(h_enc, self.W_hsigma) + self.b_hsigma
        self.var = T.exp(log_var)
        self.sigma = T.sqrt(self.var)

        srng = T.shared_randomstreams.RandomStreams(234)
        eps = srng.normal(self.mu.shape)
        self.z = self.mu + self.sigma * eps

        # decoder
        h_dec = T.nnet.relu(T.dot(self.z, self.W_zh) + self.b_zh)
        if self.continuous:
            self.reconstruct = T.dot(h_dec, self.W_hyu) + self.b_hyu
            self.log_var_dec = T.dot(h_dec, self.W_hysigma) + self.b_hysigma
            self.var_dec = T.exp(self.log_var_dec)
        else:
            self.reconstruct = T.nnet.sigmoid(T.dot(h_dec, self.W_hy) + self.b_hy)

    def multivariate_bernoulli(self, y_pred, y_true):
        return T.sum(y_true * T.log(y_pred) + (1 - y_true) * T.log(1 - y_pred), axis=1)
   
    def log_mvn(self, y_pred, y_true):
        p = y_true.shape[1]
        return T.sum(-0.5 * p * np.log(2 * np.pi) - 0.5 * self.log_var_dec - 0.5 * ((y_true - y_pred)**2 / self.var_dec), axis=1)

    def kld(self, mu, var):
        return 0.5 * T.sum(1 + T.log(var) - mu**2 - var, axis=1)

    def define_train_test_funcs(self):
        if self.continuous:
            cost = -T.mean((self.kld(self.mu, self.var) + self.log_mvn(self.reconstruct, self.X)))
        else:
            cost = -T.mean((self.kld(self.mu, self.var) + self.multivariate_bernoulli(self.reconstruct, self.X))) 
        
        gparams = []
        for param in self.params:
            #gparam = T.grad(cost, param)
            gparam = T.clip(T.grad(cost, param), -10, 10)
            gparams.append(gparam)

        lr = T.scalar("lr")
        optimizer = eval(self.optimizer)
        updates = optimizer(self.params, gparams, lr)
        
        self.train = theano.function(inputs = [self.X, lr], outputs = [cost, self.z], updates = updates)
        self.validate = theano.function(inputs = [self.X], outputs = [cost, self.reconstruct])
        self.project = theano.function(inputs = [self.X], outputs = self.mu)
        self.generate = theano.function(inputs = [self.z], outputs = self.reconstruct)
  
