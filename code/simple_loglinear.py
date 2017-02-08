##!/usr/bin/env python
import numpy as np
from optimizers import sgd
from optimizers import momentum
import theano
import theano.tensor as T


__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64

class SimpleLoglinear(object):
    def __init__(self, dh, reg = 0.1, learner_reg = 0.5, adapt = False, x1 = 1.0, x2 = 0.2, clip = False):
        self.dh = dh #DataHelper(event2feats_file, feat2id_file, actions_file)
        self.adapt = adapt
        self.clip = clip
        self.l = reg #reg parameter
        self.ul = learner_reg #reg parameter for the learner model
        self._eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self._mult_eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
        if self.adapt:
            x = 5 * np.ones((self.dh.FEAT_SIZE,))
            self.b_z = theano.shared(floatX(x), name='b_z')
            self.b_r = theano.shared(floatX(x), name='b_z')
            self.W_r = theano.shared(floatX(x1 * x), name='W_r')
            self.W_z = theano.shared(floatX(x2 * x), name='W_z')
            self.params = [self.W_r, self.b_r, self.W_z, self.b_z]
        else:
            x = 1.0
            self.W_r = theano.shared(floatX(x1 * x), name='W_r')
            self.W_z = theano.shared(floatX(x2 * x), name='W_z')
            self.params = [self.W_r, self.W_z]
        self.reg_params = [self.W_r, self.W_z]
        self.make_graph()

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

    def make_graph(self):
        lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        X = T.ivector('X') #(sequence_size,) #index of the input string
        O = T.fmatrix('O') #(sequence_size, output_dim) #mask for possible Ys
        Y = T.fmatrix('Y') #(sequence_size, output_dim) #the Y that is selected by the user
        S = T.fmatrix('S') #(sequence_size,7) # was the answer marked as correct or incorrect?
        theta_0 = T.fvector('theta_0') #(feature_size,)

        def rms(v):
            return T.sqrt(v + self._eps)

        def bin_loss(y_selected, y_predicted, feedback):
            yp_unsafe = T.switch(y_selected,y_predicted,0.0)
            yp = T.clip(yp_unsafe, self._eps, 0.9999999)
            l = T.switch(feedback[4,7], -T.log(yp), -T.log(1.0 - yp))
            l = T.switch(feedback[3], 0, l)
            return l

        def create_target(y_selected, y_predicted, feedback):
            #y_selected is a one-hot vector
            #y_predicted is the output based on current set of weights
            #feedback is the indication that the system gives the user i.e. correct or incorrect.
            y_neg_selected = -(y_selected - 1.0)  #flips 1s to 0s and 0s to 1s
            y_rev_predicted = y_predicted * y_neg_selected
            y_rev_predicted = y_rev_predicted / y_rev_predicted.sum(axis=1)[:, np.newaxis]
            y_target = T.switch(feedback[3], y_selected, y_rev_predicted) #if answer revealed then y_selected else it can be correct or incorrect
            y_target = T.switch(feedback[4], y_predicted, y_target) #if correct return predicted, else return target (which is now incorrect)
            return y_target

        def masked(a, mask, val):
            a_m = T.switch(mask, a, val)
            return a_m

        def assign_losses(loss_t, s_t):
            r_loss_t = T.switch(s_t[3], loss_t, 0)
            c_loss_t = T.switch(T.any(s_t[[4,7]]), loss_t, 0)
            ic_loss_t = T.switch(T.any(s_t[[5,8]]), loss_t, 0)
            return r_loss_t, c_loss_t, ic_loss_t

        def log_linear_t(Phi_x_t, y_t, o_t, s_t, theta_t):
            y_dot = Phi_x_t.dot(theta_t.T) #(Y,D,) dot (D,)
            y_dot_masked = masked(y_dot, o_t, -100000000.0000) #(Y,)
            y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(Y,)
            y_hat = T.clip(y_hat_unsafe, self._eps, 0.9999999)
            model_loss = -T.sum(y_t * T.log(y_hat))
            y_target = create_target(y_t, y_hat, s_t)
            #user_loss = -T.sum(y_target * T.log(y_hat), axis=1) + self.ul * T.sum(T.sqr(theta_t))
            #theta_t_grad = T.grad(user_loss, theta_t)
            theta_t_grad = y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t)  - (self.ul * 2.0 * theta_t) #obs - exp
            theta_t_grad = T.reshape(theta_t_grad, theta_t.shape) #(D,)
            y_hat = T.reshape(y_hat, (self.dh.E_SIZE,)) #(Y,)
            return theta_t_grad, y_hat, model_loss

        def recurrence(x_t, y_t, o_t, s_t, theta_t):
            #x_t (scalar)
            #y_t (Y,)
            #o_t (Y,)
            #s_t (9,)
            #theta_t (D,)
            s_t = T.reshape(s_t, (9,))
            theta_t = T.reshape(theta_t, (self.dh.FEAT_SIZE,))
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            grad_theta_t, y_hat, loss_t = log_linear_t(Phi_x_t, y_t, o_t, s_t, theta_t) #(D,) and scalar
            r_loss_t, c_loss_t, ic_loss_t = assign_losses(loss_t, s_t)
            if s_t[6] == 1: #nofeeback no knowledge change...
                theta_tp1 = theta_t
            else:
                if self.adapt:
                    theta_tp1 = T.nnet.sigmoid(self.W_r + self.b_r) * theta_t + T.nnet.sigmoid(self.W_z + self.b_z) * grad_theta_t
                else:
                    theta_tp1 = T.nnet.sigmoid(self.W_r) * theta_t + T.nnet.sigmoid(self.W_z) * grad_theta_t

            if self.clip:
                theta_tp1 = T.clip(theta_tp1, -1.0, 1.0)
            else:
                pass
            return theta_tp1, y_hat, loss_t, r_loss_t,c_loss_t, ic_loss_t

        [seq_thetas, seq_y_hats, all_losses, r_losses, c_losses, ic_losses], _ = theano.scan(fn=recurrence, 
                sequences=[X,Y,O,S], 
                outputs_info=[theta_0, None, None, None, None, None])

        all_loss = T.sum(all_losses)
        r_loss = T.mean(T.nonzero_values(r_losses))
        c_loss = T.mean(T.nonzero_values(c_losses))
        ic_loss = T.mean(T.nonzero_values(ic_losses))
        reg_loss = 0.0
        for reg_param in self.reg_params:
            reg_loss += T.sum(T.abs_(reg_param + self._eps))
            reg_loss += T.sum(T.sqr(reg_param))
        total_loss = ic_loss + (self.l * reg_loss)
        self.get_seq_loss = theano.function([X, Y, O, S, theta_0], outputs = all_loss)
        self.get_params = theano.function(inputs = [], outputs = self.params)
        self.get_seq_losses = theano.function([X, Y, O, S, theta_0], outputs = [all_losses, r_losses, c_losses, ic_losses])
        self.get_loss = theano.function([X, Y, O, S, theta_0], outputs = [total_loss, all_loss, r_loss, c_loss, ic_loss])
        self.get_seq_y_hats = theano.function([X, Y, O, S, theta_0], outputs = seq_y_hats)
        self.get_seq_thetas = theano.function([X, Y, O, S, theta_0], outputs = seq_thetas)
        self.do_sgd_update = theano.function([X, Y, O, S, theta_0, lr], 
                outputs= [total_loss, seq_thetas, seq_y_hats], 
                updates = momentum(total_loss, self.params, lr))
        self.do_rms_update = theano.function([X, Y, O, S, theta_0, lr], 
                outputs= [total_loss, seq_thetas, seq_y_hats], 
                updates = sgd(total_loss, self.params, lr))
