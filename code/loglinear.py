##!/usr/bin/env python
import numpy as np
from my_utils import rargmax
import theano
import theano.tensor as T
from datahelper import DataHelper

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64

def _get_zeros(name, *shape, **kwargs):
    return theano.shared(np.ones(shape=shape, dtype=floatX), name=name, borrow=True)

class UserModel(object):
    def __init__(self, event2feats_file, feat2id_file, actions_file, optimizer=None):
        self.dh = DataHelper(event2feats_file, feat2id_file, actions_file)
        self.optimizer = optimizer
        self.reg_type = 'l2'
        self.l = 0.01 # regularization parameter
        self._eps = 1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self.decay = 0.9 # for RMSprop
        W = _get_zeros("W", 1, self.dh.FEAT_SIZE) #theano.shared(floatX(w), name='W')
        b = _get_zeros("b", 1, self.dh.E_SIZE) #theano.shared(floatX(w), name='W')
        self.params = [W, b]
        self.reg_params = [W]
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi')
        self.__theano_init__()

    def __theano_init__(self):
        #lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        x = T.lvector('x') #(batch_size,) #index of the input string
        o = T.fmatrix('o') #(batch_size, output_dim) #mask for possible Ys
        y_selected = T.fmatrix('y_selected') #(batch_size, output_dim) #the Y that is selected by the user
        f = T.lvector('f') #(batch_size,) # was the answer marked as correct or incorrect?
        reg_l2 = 0.0
        reg_l1 = 0.0
        for w in self.reg_params:
            reg_l2 += T.sum(T.sqr(w)) 
            reg_l1 += T.sum(abs(w))

        y_dot = self.phi[x,:,:].dot(self.params[0].T)[:,:,0] + self.params[1] # perform dot product of Phi(x) and self.weights
        y_dot_masked = self.masked(y_dot, o, -np.inf) #(batch_size, output_dim)
        y_hat  = T.nnet.softmax(y_dot_masked) #(batch_size, output_dim)
        y_target = self.create_target(y_selected, y_hat, f)
        loss_vec = -T.sum(y_target * T.log(y_hat + self._eps), axis=1) #(batch_size,) #cross-entropy

        loss = T.mean(loss_vec) + (self.l * (reg_l2 + reg_l1))

        dW = [T.grad(loss, p) for p in self.params]
        self.y_dot_x = theano.function([x], y_dot)
        self.y_dot_masked_x = theano.function([x, o], y_dot_masked)
        self.y_given_x = theano.function([x, o], y_hat)
        self.y_target_x = theano.function([x, o, y_selected, f], y_target)

        self.get_loss = theano.function(inputs = [x,o,y_selected,f], outputs = loss)
        self.get_grad = theano.function(inputs = [x, o, y_selected,f], outputs=dW)

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
