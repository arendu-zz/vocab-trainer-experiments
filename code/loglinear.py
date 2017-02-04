##!/usr/bin/env python
import numpy as np
from my_utils import rargmax
import theano
import theano.tensor as T
from datahelper import DataHelper

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

class UserModel(object):
    def __init__(self, event2feats_file, feat2id_file, actions_file, regularization = 0.1, reg_type = 'l2'):
        self.dh = DataHelper(event2feats_file, feat2id_file, actions_file)
        self.reg_type = reg_type 
        self.l = regularization #regularization parameter
        self._eps = np.finfo(np.float32).tiny #1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self.decay = 0.9 # for RMSprop
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi')
        self.__theano_init__()

    def __theano_init__(self):
        #lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        x = T.ivector('x') #(batch_size,) #index of the input string
        o = T.imatrix('o') #(batch_size, output_dim) #mask for possible Ys
        y_selected = T.imatrix('y_selected') #(batch_size, output_dim) #the Y that is selected by the user
        f = T.ivector('f') #(batch_size,) # was the answer marked as correct or incorrect?
        W = T.fvector('W') #(1, feature_size)
        if self.reg_type == 'l2':
            reg = T.sum(T.sqr(W)) 
        elif self.reg_type == 'l1':
            reg = T.sum(T.abs_(W + self._eps))
        elif self.reg_type == 'elastic':
            reg = T.sum(T.sqr(W)) + T.sum(T.abs_(W + self._eps))
        else:
            raise Exception("unknown reg_type")

        y_dot = self.phi[x,:,:].dot(W.T) 
        #y_dot = self.phi[x,:,:].dot(self.W.T)[:,:,0] + self.b
        y_dot_masked = self.masked(y_dot, o, -np.inf) #(batch_size, output_dim)
        y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(batch_size, output_dim)
        y_hat = T.clip(y_hat_unsafe, self._eps, 0.9999999)
        y_target = self.create_target(y_selected, y_hat, f)
        loss_vec = -T.sum(y_target * T.log(y_hat + self._eps), axis=1) #(batch_size,) #cross-entropy

        loss = T.mean(loss_vec) #+ (self.l * reg) 
        my_grad = T.reshape(-(y_target.dot(self.phi[x,:,:]) - y_hat.dot(self.phi[x,:,:])), W.shape)

        #g_params = T.grad(loss, W)
        self.y_dot_x = theano.function([W, x], y_dot)
        self.get_reg = theano.function([W], reg)
        #self.get_phi_x = theano.function([x], self.phi[x,:,:])
        self.y_dot_masked_x = theano.function([W, x, o], y_dot_masked)
        self.y_given_x = theano.function([W, x, o], y_hat)
        #self.y_given_x_unmasked = theano.function([W, x, o], y_hat_unmasked)
        self.y_target_x = theano.function([W, x, o, y_selected, f], y_target)
        self.get_loss = theano.function(inputs = [W, x, o, y_selected,f], outputs=loss)
        #self.get_grad = theano.function(inputs = [W, x, o, y_selected,f], outputs=g_params)
        self.get_grad = theano.function(inputs = [W, x, o, y_selected,f], outputs=my_grad)
        #self.get_next_params = theano.function(inputs = [W, x, o, y_selected, f, lr], outputs=[W, b], 
        #                                     updates = [(p, p - lr * g) for p,g in zip([W,b], g_params)])

        #self.do_sgd_update = theano.function(inputs =[x, y_selected,f, lr], outputs=[loss, loss_vec], 
        #        updates = [(self.weights, self.weights - (lr * dW))])

    def _phi(self, f_idx, e_idx):
        ff = np.zeros(self.dh.FEAT_SIZE)
        ff_idx, ff_vals = self.dh.event2feats[f_idx, e_idx]
        ff[ff_idx] = ff_vals
        return ff

    def load_phi(self):
        p = np.zeros((self.dh.F_SIZE, self.dh.E_SIZE, self.dh.FEAT_SIZE))
        for f_idx in xrange(self.dh.F_SIZE):
            for e_idx in xrange(self.dh.E_SIZE):
                p[f_idx, e_idx, :]  = self._phi(f_idx, e_idx)
        return p

    def RMS(self, v):
        return T.sqrt(v + 1e-8) 

    def create_target(self, y_selected, y_predicted, feedback):
        #y_selected is a one-hot vector
        #y_predicted is the output based on current set of weights
        #feedback is the indication that the system gives the user i.e. correct or incorrect.
        y_neg_selected = -(y_selected - 1.0)  #flips 1s to 0s and 0s to 1s
        y_predicted = y_predicted * y_neg_selected
        y_predicted = y_predicted / y_predicted.sum(axis=1)[:, np.newaxis]
        y_target = T.switch(feedback[:, None], y_selected, y_predicted)
        return y_target

    def masked(self, a, mask, val):
        a_m = T.switch(mask, a, val)
        return a_m

    def masked_softmax(self, a, mask, axis):
        e_a = T.exp(a)
        e_m = T.switch(mask, e_a, self._eps)
        sum_e_m = T.sum(e_m, axis, keepdims = True)
        r = e_m / sum_e_m
        return r
