##!/usr/bin/env python
import numpy as np
from my_utils import rargmax
import theano
import pdb
import theano.tensor as T
from datahelper import DataHelper

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64

class UserModel(object):
    RMSProp = 'rmsprop'
    SGD = 'sgd'

    def __init__(self, event2feats_file, feat2id_file, actions_file, optimizer = None, extra_state_params = None):
        self.dh = DataHelper(event2feats_file, feat2id_file, actions_file)
        self.optimizer = optimizer
        self.extra_state_params = extra_state_params
        self.reg_type = 'l2'
        self.l = 0.01
        self.lr = 0.1
        self._eps = 1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self.decay = 0.9 # for RMSprop
        self.last_seen_f_id = 0
        w = np.zeros((1,self.dh.FEAT_SIZE))
        self.weights = theano.shared(floatX(w), name='W')
        self.E_gW = theano.shared(floatX(np.zeros_like(w)), name='E_gW')
        #self.E_g = np.zeros_like(self.weights)
        #self.E_dx = np.zeros_like(self.weights)
        self.seen_actions = set([])
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi')
        self.__theano_init__()

    def get_properties(self):
        return 'Theano_usermodel properties:', 'reg type:', self.reg_type, 'reg_param:',self.l, 'weight update method:', self.optimizer, 'eta:', self._eta, 'decay:',self.decay

    def get_state(self):
        w = self.weights.get_value()
        if self.extra_state_params:
            y_g_x = self.y_given_x(np.array([self.last_seen_f_id]))
            w = np.append(w, y_g_x, axis = 1)
        assert w.shape == (1, self.state_size())
        return floatX(w)

    def reset(self):
        w = np.zeros((1,self.dh.FEAT_SIZE))
        self.weights.set_value(floatX(w)) 
        return True

    def weight_size(self):
        return self.weights.get_value().shape

    def state_size(self):
        r = self.weights.get_value().shape[1] + (len(self.dh.e2id) if self.extra_state_params else 0)
        return int(r)

    def set_weights(self, w):
        if w is not None:
            self.weights.set_value(floatX(w))
        else:
            pass
        return True

    def action_size(self):
        return len(self.dh.actions)

    def load_weights(self, weights_path):
        raise NotImplementedError("Weights should be loaded from a saved file...")

    def get_response(self, f_id,e_id, o_ids, response_strategy):
        x = np.array([f_id])
        y_score_x = self.y_score_x(x)
        m = np.zeros_like(y_score_x)
        m[0,o_ids] = 1.0
        y_score_x = y_score_x * m
        if response_strategy == 'max':
            s_id = rargmax(y_score_x[0]) 
        else:
            raise NotImplementedError('only max response for now...')
        return s_id

    def update(self, action, response_strategy, observation = None):
        assert type(action) == tuple
        f_id = self.dh.f2id[action[2]]
        e_id = self.dh.e2id[action[3]]
        x = np.array([f_id])
        y = np.zeros((1, self.dh.E_SIZE))
        y[0, e_id] = 1.0
        if action[1] == 'EX':
            if self.optimizer is None or self.optimizer == UserModel.SGD:
                o_hot = np.ones_like(y) 
                self.do_sgd_update_mc_c(x, floatX(y), floatX(o_hot), floatX(self.lr))
            elif self.optimizer == UserModel.RMSProp:
                o_hot = np.ones_like(y) 
                self.do_rmsprop_update_mc_c(x, floatX(y), floatX(o_hot))
            else:
                raise BaseException("unknown optimization method..")
        else:
            o_ids = [self.dh.e2id[o] for o in action[3:]]
            o_hot = np.zeros_like(y) 
            o_hot[0, o_ids] = 1
            o_hot = floatX(o_hot)
            if observation is None:
                s_id  = self.sample_response(f_id, e_id, o_ids, response_strategy)
            else:
                s_id = observation
            if s_id == e_id: # correct guess
                if self.optimizer is None or self.optimizer == UserModel.SGD:
                    self.do_sgd_update_mc_c(x, floatX(y), floatX(o_hot), floatX(self.lr))
                elif self.optimizer == UserModel.RMSProp:
                    self.do_rmsprop_update_mc_c(x, floatX(y), floatX(o_hot))
                else:
                    raise BaseException("unknown optimization method..")
            else: # incorrect guess
                y_score = self.y_score_x(x)
                y_score *= o_hot
                y_score[0, s_id] = 0.0
                y_redistributed = y_score / np.sum(y_score, axis=1, keepdims = True)
                #u_hot = np.ones_like(y)
                #u_hot *= o_hot
                #u_hot[0, s_id] = 0.0
                if self.optimizer is None or self.optimizer == UserModel.SGD:
                    self.do_sgd_update_mc_c(x, floatX(y_redistributed), floatX(o_hot), floatX(self.lr))
                    #self.do_sgd_update_mc_ic(x, floatX(u_hot), floatX(o_hot), floatX(self.lr))
                elif self.optimizer == UserModel.RMSProp:
                    self.do_rmsprop_update_mc_c(x, floatX(y_redistributed), floatX(o_hot))
                    #self.do_rmsprop_update_mc_ic(x, floatX(u_hot), floatX(o_hot))
                else:
                    raise BaseException("unknown optimization method..")
        return True 

    def get_y_score_x(self, f_id, w = None):
        self.set_weights(w)
        x = np.array([f_id])
        return self.y_score_x(x)[0]

    def get_y_given_x(self, f_id, w = None):
        self.set_weights(w)
        x = np.array([f_id])
        return self.y_given_x(x)[0]

    def __theano_init__(self):
        lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        x = T.lvector('x') #(batch_size,)
        y = T.fmatrix('y') #(batch_size, output_dim)
        o = T.fmatrix('o') #(batch_size, output_dim)
        u = T.fmatrix('u') #(batch_size, output_dim)
        reg_l2 = 0.5 * T.sum(T.sqr(self.weights)) 

        y_dot = self.phi[x,:,:].dot(self.weights.T)[:,:,0]
        y_score= T.exp(y_dot)
        y_hat  = T.nnet.softmax(y_dot) #(batch_size, output_dim)
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
