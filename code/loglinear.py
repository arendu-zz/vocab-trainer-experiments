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

class UserModel(object):
    def __init__(self, event2feats_file, feat2id_file, actions_file, optimizer=None):
        self.dh = DataHelper(event2feats_file, feat2id_file, actions_file)
        self.optimizer = optimizer
        self.reg_type = 'l2'
        self.l = 0.01
        self.lr = 0.1
        self._eps = 1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self.decay = 0.9 # for RMSprop
        self.last_seen_f_id = 0
        self.W = self._get_zeros("W", 1, self.dh.FEAT_SIZE) #theano.shared(floatX(w), name='W')
        self.b = self._get_zeros("b", 1, self.dh.FEAT_SIZE) #theano.shared(floatX(w), name='W')
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi')

    def _get_zeros(name, *shape, **kwargs):
        return theano.shared(np.zeros(shape=shape, dtype=floatX), name=name, borrow=True)

    def __theano_init__(self):
        lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        x = T.lvector('x') #(batch_size,) #index of the input string
        y = T.fmatrix('y') #(batch_size, output_dim) #distribution over Y
        o = T.fmatrix('o') #(batch_size, output_dim) #mask for possible Ys
        f = T.fmatrix('f') #(batch_size,) # was the answer marked as correct or incorrect?
        #u = T.fmatrix('u') #(batch_size, output_dim)
        reg_l2 = 0.5 * T.sum(T.sqr(self.weights)) 

        y_dot = self.phi[x,:,:].dot(self.weights.T)[:,:,0] # perform dot product of Phi(x) and self.weights
        y_score = T.exp(y_dot)
        y_hat  = T.nnet.softmax(y_dot) #(batch_size, output_dim)
        y_hat_masked = self.masked_softmax(y_dot, o, 1) #(batch_size, output_dim)
        y_true = _get_zeros('y_true', 1, self.dh.f

        #loss_vec = -T.sum(y * T.log(y_hat), axis=1) #(batch_size,)
        #loss = T.mean(loss_vec) + self.l * reg_l2

        #y_dot_o = T.switch(o, y_dot,np.NINF) 
        y_hat_mc_o = self.masked_softmax(y_dot, o, 1) #T.nnet.softmax(y_dot_o)
        loss_vec_mc_c = -T.sum(y * T.log(y_hat_mc_o), axis = 1) 
        loss_mc_c = T.mean(loss_vec_mc_c) + self.l * reg_l2

        #y_dot_u = T.switch(u, y_dot, np.NINF)
        y_hat_mc_ic = self.masked_softmax(y_dot, u, 1) #T.nnet.softmax(y_dot_u) 
        loss_vec_mc_ic = -T.sum(y_hat_mc_ic * T.log(y_hat_mc_o), axis = 1) 
        loss_mc_ic = T.mean(loss_vec_mc_ic) + self.l * reg_l2

        #dW = T.grad(loss, self.weights)
        dW_mc_c = T.grad(loss_mc_c, self.weights)
        E_gW = (self.decay * self.E_gW) + ((1.0 - self.decay) * T.sqr(dW_mc_c)) 
        rms_dW_mc_c = (self._eta / self.RMS(E_gW)) * dW_mc_c  #T.grad(loss_mc_c, self.weights)

        dW_mc_ic = T.grad(loss_mc_ic, self.weights)
        E_gW_ic = (self.decay * self.E_gW) + ((1.0 - self.decay) * T.sqr(dW_mc_ic)) 
        rms_dW_mc_ic = (self._eta / self.RMS(E_gW_ic)) * dW_mc_ic  #T.grad(loss_mc_c, self.weights)

        self.get_weights = theano.function(inputs = [], outputs = self.weights)
        self.y_score_x = theano.function([x], y_score)
        self.y_given_x = theano.function([x], y_hat)

        #self.get_loss = theano.function(inputs = [x, y], outputs = loss)
        #self.get_loss_vec = theano.function(inputs = [x, y], outputs = loss_vec)
        self.get_loss_mc_c = theano.function(inputs = [x, y, o], outputs = loss_mc_c)
        self.get_loss_mc_ic = theano.function(inputs = [x, u, o], outputs = loss_mc_ic)
        self.get_loss_vec_mc_c = theano.function(inputs = [x, y, o], outputs = loss_vec_mc_c)
        self.get_loss_vec_mc_ic = theano.function(inputs = [x, u, o], outputs = loss_vec_mc_ic)

        #self.get_grad = theano.function(inputs = [x, y], outputs=dW)
        self.get_grad_mc_c = theano.function(inputs = [x, y, o], outputs=dW_mc_c)
        self.get_grad_mc_ic = theano.function(inputs = [x, u, o], outputs=dW_mc_ic)

        #self.do_sgd_update = theano.function(inputs =[x, y, lr], outputs=[loss, loss_vec], 
        #        updates = [(self.weights, self.weights - (lr * dW))])

        self.do_sgd_update_mc_c = theano.function(inputs =[x, y, o, lr], outputs=[loss_mc_c, loss_vec_mc_c], 
                updates = [(self.weights, self.weights - (lr * dW_mc_c))])

        self.do_rmsprop_update_mc_c = theano.function(inputs =[x, y, o], 
                outputs=[loss_mc_c, loss_vec_mc_c], 
                updates = [(self.weights, self.weights - rms_dW_mc_c)] + [(self.E_gW, E_gW)])

        self.do_sgd_update_mc_ic = theano.function(inputs =[x, u, o, lr], outputs=[loss_mc_ic, loss_vec_mc_ic], 
                updates = [(self.weights, self.weights - (lr * dW_mc_ic))])

        self.do_rmsprop_update_mc_ic = theano.function(inputs =[x, u, o], 
                outputs=[loss_mc_ic, loss_vec_mc_ic], 
                updates = [(self.weights, self.weights - rms_dW_mc_ic)] + [(self.E_gW, E_gW_ic)])

        #tmp functions
        self.get_y_hat_mc_o = theano.function([x,o], y_hat_mc_o)
        self.get_y_hat_mc_ic = theano.function([x, u], y_hat_mc_ic)
        self.get_y_dot = theano.function([x], y_dot)
        #self.get_y_dot_o = theano.function([x,o], y_dot_o)
        #self.get_y_dot_u = theano.function([x,u], y_dot_u)

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

    def masked_softmax(self, a, mask, axis):
        e_a = T.exp(a)
        e_m = T.switch(mask, e_a, self._eps)
        sum_e_m = T.sum(e_m, axis, keepdims = True)
        r = e_m / sum_e_m
        return r
