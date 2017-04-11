#!/usr/bin/env python
import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T
import theano.tensor.shared_randomstreams
from optimizers import sgd, rmsprop

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64


def huber(target, output):
    delta = 2.0
    d = target - output
    a = .5 * d**2
    b = delta * (T.abs_(d + 1e-8) - delta / 2.0)
    l = T.switch(T.lt(T.abs_(d + 1e-8), delta), a, b)
    return l

class Actor(object):
    def __init__(self, layer_dims, reg_param=0.1, use_target_networks = False):
        self.layer_dims = layer_dims
        self.layer_weights = []
        self.layer_bias = [] 
        self.use_target_networks = use_target_networks
        self.tau = 0.1 if self.use_target_networks else 1.0
        self.target_layer_weights = []
        self.target_layer_bias = []
        self.l = reg_param 
        self.eps = 1e-8 #np.finfo(np.float32).eps
        self.reset()
        self._update = rmsprop
        self.rng =T.shared_randomstreams.RandomStreams(seed=124)
        self.__theano_init__()

    def save(self, save_location):
        f = open(save_location, 'wb')
        pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.close()

    @staticmethod
    def load(load_location):
        f = open(load_location, 'rb')
        dqn = pickle.load(f)
        f.close()
        return dqn

    def get_properties(self):
        info  = 'network properties: structure:' + ','.join([str(i) for i in self.layer_dims])
        info += '\nreg_type: + self.reg_type' + 'reg_param:' + str(self.l)
        return info

    def reset(self):
        idx = 1
        for layer_dim in self.layer_dims[1:]:
            f_in = self.layer_dims[idx -1]
            _w = np.random.uniform(-1.0 / f_in, +1.0 / f_in, (self.layer_dims[idx - 1], self.layer_dims[idx]))
            #_w = np.random.normal(0.0, 2.0 / (self.layer_dims[idx - 1] + self.layer_dims[idx]), (self.layer_dims[idx - 1], self.layer_dims[idx]))
            #_target_w = np.random.normal(0.0, 2.0 / (self.layer_dims[idx - 1] + self.layer_dims[idx]), (self.layer_dims[idx - 1], self.layer_dims[idx]))
            W = theano.shared(_w.astype(floatX), name='W' + str(idx))
            target_W = theano.shared(_w.astype(floatX), name='target_W' + str(idx))
            _b = np.random.uniform(-0.01, 0.01, (self.layer_dims[idx],))
            b = theano.shared(_b.astype(floatX), name = 'b' + str(idx))
            target_b = theano.shared(_b.astype(floatX), name = 'target_b' + str(idx))
            self.layer_weights.append(W)
            self.layer_bias.append(b)
            self.target_layer_weights.append(target_W)
            self.target_layer_bias.append(target_b)
            idx += 1
        return True

    def __theano_init__(self):
        lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        B_t = T.fmatrix('B_t') #(batch_size, input_dim,) #(input_dim is state space)
        a_t = T.lvector('a_t') # (batch_size,)
        r_t = T.fvector('r_t') # (batch_size,)

        def forward_pass(input_state, on_target = False):
            activation = None
            dot_prod = None
            a_hat = None
            weights = self.target_layer_weights if on_target else self.layer_weights
            bias = self.target_layer_bias if on_target else self.layer_bias
            for idx, (W,b) in enumerate(zip(weights, bias)):
                if idx == 0:
                    dot_prod = T.dot(input_state, W) + b 
                else:
                    dot_prod = T.dot(activation, W) + b 

                if idx == len(self.layer_weights) - 1:
                    a_hat = T.nnet.softmax(dot_prod) #last linear layer
                else:
                    activation = T.nnet.relu(dot_prod, 0.1)  # all other layers are relu activated
                    #activation = T.tanh(dot_prod)  # all other layers are relu activated
            return a_hat #(batch_size, action_space)

        a_hat = forward_pass(B_t, on_target = self.use_target_networks)
        #xent = T.nnet.categorical_crossentropy(a_hat, a_t)
        xent = T.log(a_hat[T.arange(a_hat.shape[0]), a_t] + 1e-8)
        #loss_vec = -T.log(a_hat[a_t])
        loss_vec = xent * r_t
        batch_loss = -T.mean(loss_vec)

        reg_l2 = 0.0 
        for wts in self.layer_weights:
            reg_l2 += T.sum(T.sqr(wts)) 

        total_loss = batch_loss + (self.l * reg_l2)

        self.get_a_hat = theano.function([B_t], a_hat)
        self.get_reg = theano.function(inputs=[], outputs=reg_l2)
        self.get_loss_vec = theano.function([B_t, a_t, r_t], outputs=[total_loss, batch_loss, loss_vec])
        self.target_update = theano.function([], [], updates=[(tp, self.tau * p + (1.0 - self.tau) * tp) for tp,p in zip(self.target_layer_weights + self.target_layer_bias, self.layer_weights + self.layer_bias)])
        self.do_update = theano.function([B_t, a_t, r_t, lr], 
                outputs= [total_loss, batch_loss, loss_vec], 
                updates = self._update(total_loss, self.layer_weights + self.layer_bias, lr))
