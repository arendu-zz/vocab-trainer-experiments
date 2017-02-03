##!/usr/bin/env python
import numpy as np
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

def rmsprop(cost, params, learning_rate, rho=0.9, epsilon=1e-6):
    updates = list()

    for param in params:
        accu = theano.shared(np.zeros(param.get_value(borrow=True).shape, dtype=floatX),
                             broadcastable=param.broadcastable)

        grad = T.grad(cost, param)
        accu_new = rho * accu + (1 - rho) * grad ** 2

        updates.append((accu, accu_new))
        updates.append((param, param - (learning_rate * grad / T.sqrt(accu_new + epsilon))))

    return updates

def sgd(cost, params, learning_rate):
    return [(param, param - learning_rate * T.grad(cost, param)) for param in params]


class GatedLoglinear(object):
    def __init__(self, event2feats_file, feat2id_file, actions_file, regularization = 0.1, reg_type = 'l2'):
        self.dh = DataHelper(event2feats_file, feat2id_file, actions_file)
        self.reg_type = reg_type 
        self.l = regularization #regularization parameter
        self.lr = 0.1
        self._eps = np.finfo(np.float32).tiny #1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self.decay = 0.9 # for RMSprop
        self.W_zw = theano.shared(floatX(0.001 * np.ones((self.dh.FEAT_SIZE, self.dh.FEAT_SIZE))), name='W_zw')
        self.W_zx = theano.shared(floatX(0.001 * np.ones((1, self.dh.E_SIZE))), name='W_zx')
        self.b_z = theano.shared(floatX(0.001 * np.ones((1, self.dh.FEAT_SIZE))), name='b_z')

        self.W_rw = theano.shared(floatX(0.001 * np.ones((self.dh.FEAT_SIZE, self.dh.FEAT_SIZE))), name='W_rw')
        self.W_rx = theano.shared(floatX(0.001 * np.ones((1, self.dh.E_SIZE))), name='W_rx')
        self.b_r = theano.shared(floatX(0.001 * np.ones((1, self.dh.FEAT_SIZE))), name='b_r')
        self.params = [self.W_zw, self.W_zx, self.b_z, self.W_rw, self.W_rx, self.b_r]
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
        self.__theano_init__()

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

    def __theano_init__(self):
        #lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        X = T.ivector('X') #(sequence_size,) #index of the input string
        O = T.imatrix('O') #(sequence_size, output_dim) #mask for possible Ys
        Y = T.imatrix('Y') #(sequence_size, output_dim) #the Y that is selected by the user
        F = T.ivector('F') #(sequence_size,) # was the answer marked as correct or incorrect?
        theta_0 = T.fvector('theta_0') #(feature_size,)

        def create_target(y_selected, y_predicted, feedback):
            #y_selected is a one-hot vector
            #y_predicted is the output based on current set of weights
            #feedback is the indication that the system gives the user i.e. correct or incorrect.
            y_neg_selected = -(y_selected - 1.0)  #flips 1s to 0s and 0s to 1s
            y_predicted = y_predicted * y_neg_selected
            y_predicted = y_predicted / y_predicted.sum(axis=1)[:, np.newaxis]
            y_target = T.switch(feedback[:, None], y_selected, y_predicted)
            return y_target

        def masked(a, mask, val):
            a_m = T.switch(mask, a, val)
            return a_m

        def log_linear_grad_t(x_t, y_t, o_t, f_t, theta_t):
            reg = T.sum(T.sqr(theta_t)) 
            y_dot = self.phi[x_t,:,:].dot(theta_t.T) 
            y_dot_masked = masked(y_dot, o_t, -np.inf) #(batch_size, output_dim)
            y_hat  = T.nnet.softmax(y_dot_masked) #(batch_size, output_dim)
            y_target = create_target(y_t, y_hat, f_t)
            ll_loss_vec = -T.sum(y_target * T.log(y_hat + self._eps), axis=1)  #cross-entropy
            ll_loss = T.mean(ll_loss_vec) + (self.l * reg) 
            theta_t_grad = T.grad(ll_loss, theta_t)
            return theta_t_grad, ll_loss

        def recurrence(x_t, y_t, o_t, f_t, theta_t):
            Phi_x_t = self.phi[x_t, :, :]
            z_t = T.nnet.sigmoid(self.W_zw.dot(theta_t) + self.W_zx.dot(Phi_x_t) + self.b_z)
            r_t = T.nnet.sigmoid(self.W_rw.dot(theta_t) + self.W_rx.dot(Phi_x_t) + self.b_r)
            grad_theta_tm1,loss_tm1 = log_linear_grad_t(Phi_x_t, y_t, o_t, f_t, theta_t)
            theta_tp1 = r_t * theta_t + z_t * grad_theta_tm1
            return theta_tp1, loss_tm1
        
        [thetas, losses], _ = theano.function(fn=recurrence, sequences=[X,Y,O,F], outputs_info=[theta_0, None])
        seq_loss = T.sum(losses)
        w_updates = rmsprop(seq_loss, self.params, self.lr)
        self.train = theano.function([X,Y,O,F], seq_loss, updates = w_updates)

