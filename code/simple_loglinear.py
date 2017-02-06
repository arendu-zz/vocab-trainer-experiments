##!/usr/bin/env python
import numpy as np
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
    def __init__(self, dh, learning_rate, regularization = 0.1, reg_type = 'l2', adapt = "adagrad"):
        self.dh = dh #DataHelper(event2feats_file, feat2id_file, actions_file)
        self.reg_type = reg_type
        self.l = regularization #regularization parameter
        self.lr = learning_rate
        self._eps = np.finfo(np.float32).tiny #1e-10 # for fixing divide by 0
        self._mult_eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self._decay = 0.9 # for RMSprop
        self.adapt = adapt 
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
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
        X = T.ivector('X') #(sequence_size,) #index of the input string
        O = T.imatrix('O') #(sequence_size, output_dim) #mask for possible Ys
        Y = T.imatrix('Y') #(sequence_size, output_dim) #the Y that is selected by the user
        F = T.ivector('F') #(sequence_size,) # was the answer marked as correct or incorrect?
        theta_0 = T.fvector('theta_0') #(feature_size,)
        E_g_0 = T.fvector('E_g_0') #(feature_size,)
        E_dx_0 = T.fvector('E_dx_0') #(feature_size,)

        def rms(v):
            return T.sqrt(v + self._eps)

        def adapt_grad(g, E_g, E_dx):
            if self.adapt == "adadelta":
                E_g = (self._decay * E_g) + ((1.0 - self._decay) * T.sqr(g))
                #E_g[T.abs_(E_g) < self._mult_eps] = 0.0
                grad = (rms(E_dx) / rms(E_g)) * g
                #grad[T.abs_(grad) < self._mult_eps] = 0.0
                E_dx = (self._decay * E_dx) + ((1.0 - self._decay) * T.sqr(grad))
                return grad, E_g, E_dx
            elif self.adapt == "adagrad":
                #g[T.abs_(g) < self._mult_eps] = 0.0
                E_g += g ** 2
                #E_g[T.abs_(E_g) < self._mult_eps] = 0.0
                grad = (self._eta / rms(E_g)) * g
                return grad, E_g, E_dx #E_dx does nothing...
            elif self.adapt == "rmsprop":
                E_g = (self._decay * E_g) + ((1.0 - self._decay) * T.sqr(g))
                #E_g[T.abs_(E_g) < self._mult_eps] = 0.0
                grad = (self._eta / rms(E_g)) * g
                return grad, E_g, E_dx
            elif self.adapt == "simple":
                grad = self.lr * g
                return grad, E_g, E_dx
            else:
                raise Exception("unknown adapt grad")
        
        def create_target(y_selected, y_predicted, feedback):
            #y_selected is a one-hot vector
            #y_predicted is the output based on current set of weights
            #feedback is the indication that the system gives the user i.e. correct or incorrect.
            y_neg_selected = -(y_selected - 1.0)  #flips 1s to 0s and 0s to 1s
            y_predicted = y_predicted * y_neg_selected
            y_predicted = y_predicted / y_predicted.sum(axis=1)[:, np.newaxis]
            y_target = T.switch(feedback, y_selected, y_predicted)
            return y_target

        def masked(a, mask, val):
            a_m = T.switch(mask, a, val)
            return a_m

        def log_linear_t(Phi_x_t, y_t, o_t, f_t, theta_t):
            #reg = T.sum(T.sqr(theta_t))  #TODO:add to gradient..
            y_dot = Phi_x_t.dot(theta_t.T) #(Y,D,) dot (D,)
            y_dot_masked = masked(y_dot, o_t, -np.inf) #(Y,)
            y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(Y,)
            y_hat = T.clip(y_hat_unsafe, self._eps, 0.9999999)
            y_target = create_target(y_t, y_hat, f_t)
            ll_loss_vec = -T.sum(y_target * T.log(y_hat), axis=1)  #cross-entropy
            ll_loss = T.mean(ll_loss_vec) #+ (self.l * reg)
            theta_t_grad = -(y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t))
            theta_t_grad = T.reshape(theta_t_grad, theta_t.shape) #(D,)
            y_hat = T.reshape(y_hat, (self.dh.E_SIZE,)) #(Y,)
            return theta_t_grad, y_hat, ll_loss

        def recurrence(x_t, y_t, o_t, f_t, theta_t, E_g_t, E_dx_t):
            #x_t (scalar)
            #y_t (Y,)
            #o_t (Y,)
            #f_t (scalar)
            #theta_t (D,)
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            grad_theta_t, y_hat, loss_t = log_linear_t(Phi_x_t, y_t, o_t, f_t, theta_t) #(D,) and scalar
            #theta_tp1 = T.clip(theta_t + self.lr * grad_theta_tm1, -1.0, 1.0)
            adapt_grad_t, E_g_tp1, E_dx_tp1 = adapt_grad(grad_theta_t, E_g_t, E_dx_t)
            #theta_tp1 = theta_t - self.lr * grad_theta_tm1
            theta_tp1 = theta_t - adapt_grad_t
            return theta_tp1, E_g_tp1, E_dx_tp1, y_hat, loss_t

        [seq_thetas, seq_E_gs, seq_E_dxs, seq_y_hats, seq_losses], _ = theano.scan(fn=recurrence, sequences=[X,Y,O,F], outputs_info=[theta_0, E_g_0, E_dx_0, None, None])
        seq_loss = T.mean(seq_losses) 
        total_loss = seq_loss #+ (self.l * reg_loss)
        self.get_seq_losses = theano.function([X, Y, O, F, theta_0, E_g_0, E_dx_0], outputs = seq_losses)
        self.get_seq_loss = theano.function([X, Y, O, F, theta_0, E_g_0, E_dx_0], outputs = seq_loss)
        self.get_total_loss = theano.function([X, Y, O, F, theta_0, E_g_0, E_dx_0], outputs = total_loss)
        self.get_seq_y_hats = theano.function([X, Y, O, F, theta_0, E_g_0, E_dx_0], outputs = seq_y_hats)
        self.get_seq_thetas = theano.function([X, Y, O, F, theta_0, E_g_0, E_dx_0], outputs = seq_thetas)
