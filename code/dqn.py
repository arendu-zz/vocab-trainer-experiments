#!/usr/bin/env python
import numpy as np
import cPickle as pickle
import theano
import theano.tensor as T
from optimizers import sgd, rmsprop

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64


class DQNTheano(object):
    def __init__(self, layer_dims, reg_type='elastic',reg_param=0.01):
        self.layer_dims = layer_dims
        self.layer_weights = []
        self.layer_bias = [] 
        self.l = reg_param 
        self.reg_type = reg_type 
        self.eps = 1e-8 #np.finfo(np.float32).eps
        self.reset()
        self._update = sgd
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
            _w = np.random.normal(0.0, 2.0 / (self.layer_dims[idx - 1] + self.layer_dims[idx]), (self.layer_dims[idx - 1], self.layer_dims[idx]))
            W = theano.shared(_w.astype(floatX), name='W' + str(idx))
            _b = np.ones(self.layer_dims[idx],)
            b = theano.shared(_b.astype(floatX), name = 'b' + str(idx))
            self.layer_weights.append(W)
            self.layer_bias.append(b)
            idx += 1
        return True

    def display_meta_params(self):
        for w in self.aa_E_gWs + self.aa_E_dWs + self.E_gWs + self.E_dWs:
            print w, w.get_value().sum()

    def __theano_init__(self):
        lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        B_t = T.fmatrix('B_t') #(batch_size, input_dim,) #(input_dim is state space)
        a_t = T.lvector('a_t') # (batch_size,)
        r_t = T.fvector('r_t') # (batch_size,)
        B_tm1 = T.fmatrix('B_tm1') #(batch_size, input_dim,) #(input_dim is state space)
        term_t = T.lvector('term_t') #(batch_size,)
        gamma = T.scalar('gamma', dtype=theano.config.floatX)
        #gamma_raised_t = T.lvector('gamma_raised_t') # (batch_size,) for discounting 

        def forward_pass(input_state):
            activation = None
            dot_prod = None
            Q_hat = None
            for idx, (W,b) in enumerate(zip(self.layer_weights, self.layer_bias)):
                if idx == 0:
                    dot_prod = T.dot(input_state, W) + b 
                else:
                    dot_prod = T.dot(activation, W) + b 

                if idx == len(self.layer_weights) - 1:
                    Q_hat = dot_prod #last linear layer
                else:
                    activation = T.nnet.relu(dot_prod)  # all other layers are relu activated
            return Q_hat #(batch_size, action_space)

        max_a_Q_t = T.max(forward_pass(B_t), axis=1) 
        target = T.switch(term_t, r_t, r_t + (gamma * max_a_Q_t)) #term_t = 1, terminal state = No future max_a_Q
        Q_tm1 = forward_pass(B_tm1) #(batch_size, action_space)
        a_t_Q_tm1 = Q_tm1[T.arange(a_t.shape[0]), a_t] #(batch_size,)
        pred_a_t = T.argmax(Q_tm1, axis=1)
        loss_vec = 0.5 * T.sqr(target - a_t_Q_tm1)
        batch_loss = T.mean(loss_vec)

        reg_l2 = 0.0 
        for wts in self.layer_weights:
            reg_l2 += T.sum(T.sqr(wts)) 

        total_loss = batch_loss + (self.l * reg_l2)

        self.get_Q_hat = theano.function([B_tm1], Q_tm1)
        self.get_reg = theano.function(inputs=[], outputs=reg_l2)
        self.get_a_t_prediction = theano.function([B_tm1], pred_a_t)
        self.get_loss_vec = theano.function([gamma, term_t, B_t, a_t, r_t, B_tm1], outputs=[loss_vec, target, a_t_Q_tm1])
        self.do_update = theano.function([gamma, term_t, B_t, a_t, r_t, B_tm1, lr], 
                outputs= [total_loss, batch_loss, loss_vec], 
                updates = self._update(total_loss, self.layer_weights + self.layer_bias, lr))
