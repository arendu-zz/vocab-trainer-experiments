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


class GatedLogLinear(object):
    def __init__(self, dh, regularization = 0.1, reg_type = 'l2', diag=False, low_rank_dim = 20, filter_loss= False):
        self.dh = dh #DataHelper(event2feats_file, feat2id_file, actions_file)
        self.reg_type = reg_type
        self.diag = diag
        self.low_rank_dim = low_rank_dim 
        self.filter_loss = filter_loss
        self.l = regularization #regularization parameter
        self.boost_user = 2.0
        self._eps = np.finfo(np.float32).tiny #1e-10 # for fixing divide by 0
        self._eta = 0.01 # for RMSprop and adagrad
        self.decay = 0.9 # for RMSprop
        self.W_zx = theano.shared(floatX(np.zeros((self.dh.E_SIZE,))), name='W_zx')
        self.W_zs = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE,6))), name='W_zs')
        self.b_z = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE,))), name='b_z')
        self.W_rx = theano.shared(floatX(np.zeros((self.dh.E_SIZE,))), name='W_rx')
        self.W_rs = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE,6))), name='W_rs')
        self.b_r = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE,))), name='b_r')

        if self.diag:
            self.W_zw_l1 = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE,))), name='W_zw_l1')
            self.W_rw_l1 = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE,))), name='W_rw_l1')
            self.params = [self.W_zw_l1, self.W_zx, self.b_z, self.W_rw_l1, self.W_rx, self.b_r]
            self.reg_params = [self.W_zw_l1, self.W_zx, self.W_rw_l1, self.W_rx] #dont regularize the bias
        else:
            self.W_zw_l1 = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE, self.low_rank_dim))), name='W_zw_l1')
            self.W_zw_l2 = theano.shared(floatX(np.zeros((self.low_rank_dim, self.dh.FEAT_SIZE))), name='W_zw_l2')
            self.W_rw_l1 = theano.shared(floatX(np.zeros((self.dh.FEAT_SIZE, self.low_rank_dim))), name='W_rw_l1')
            self.W_rw_l2 = theano.shared(floatX(np.zeros((self.low_rank_dim, self.dh.FEAT_SIZE))), name='W_rw_l2')
            self.params = [self.W_zw_l1, self.W_zw_l2, self.W_zx, self.b_z, self.W_rw_l1, self.W_rw_l2, self.W_rx, self.b_r]
            self.reg_params = [self.W_zw_l1, self.W_zw_l2, self.W_zx, self.W_rw_l1, self.W_rw_l2, self.W_rx] #dont regularize the bias
        self.params.append(self.W_zs)
        self.params.append(self.W_rs)
        self.reg_params.append(self.W_zs)
        self.reg_params.append(self.W_rs)
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
        self.make_graph()

    def load_params(self, new_params):
        assert len(self.params) == len(new_params)
        for shared_param, new_param in zip(self.params, new_params):
            shared_param.set_value(floatX(new_param))
        return True

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
        O = T.imatrix('O') #(sequence_size, output_dim) #mask for possible Ys
        Y = T.imatrix('Y') #(sequence_size, output_dim) #the Y that is selected by the user
        F = T.ivector('F') #(sequence_size,) # was the answer marked as correct or incorrect?
        S = T.fmatrix('S') #(sequence_size,6) # bit matrix containing card info and result info
        S_M1 = T.fmatrix('S_M1') #(sequence_size,6) # bit matrix containing card info and result info
        theta_0 = T.fvector('theta_0') #(feature_size,)

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
            y_dot_masked = masked(y_dot, o_t, -np.inf) #(batch_size, output_dim)
            y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(batch_size, output_dim)
            y_hat = T.clip(y_hat_unsafe, self._eps, 0.9999999)
            y_target = create_target(y_t, y_hat, f_t)
            ll_loss_vec = -T.sum(y_target * T.log(y_hat), axis=1)  #cross-entropy
            ll_loss = T.mean(ll_loss_vec) #+ (self.l * reg) 
            theta_t_grad = -(y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t))
            theta_t_grad = T.reshape(theta_t_grad, theta_t.shape)
            y_hat = T.reshape(y_hat, (self.dh.E_SIZE,))
            return theta_t_grad, y_hat, ll_loss

        def recurrence(x_t, y_t, o_t, f_t, s_t, s_tm1, theta_t):
            #x_t (scalar)
            #y_t (Y,)
            #o_t (Y,)
            #f_t (scalar)
            #s_t (6,)
            #s_tm1 (6,)
            #theta_t (D,)
            s_t = T.reshape(s_t, (6,))
            s_tm1 = T.reshape(s_tm1, (6,))
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            _tmp_z = T.reshape(self.W_zs.dot(s_t), theta_t.shape)
            _tmp_r = T.reshape(self.W_rs.dot(s_tm1), theta_t.shape)
            if self.diag:
                z_t = T.nnet.sigmoid(self.W_zw_l1 * theta_t + self.W_zx.dot(Phi_x_t) + _tmp_z + self.b_z) #(D,)
                r_t = T.nnet.sigmoid(self.W_rw_l1 * theta_t + self.W_rx.dot(Phi_x_t) + _tmp_r + self.b_r)
            else:
                z_t = T.nnet.sigmoid(self.W_zw_l1.dot(self.W_zw_l2.dot(theta_t)) + self.W_zx.dot(Phi_x_t) + _tmp_z + self.b_z) #(D,)
                r_t = T.nnet.sigmoid(self.W_rw_l1.dot(self.W_rw_l2.dot(theta_t)) + self.W_rx.dot(Phi_x_t) + _tmp_r + self.b_r)
            z_t = T.reshape(z_t, theta_t.shape) #(D,)
            r_t = T.reshape(r_t, theta_t.shape) #(D,)
            grad_theta_t, y_hat, loss_t = log_linear_t(Phi_x_t, y_t, o_t, f_t, theta_t) #(D,) and scalar
            if self.filter_loss:
                loss_t = T.switch(s_t[3], loss_t, self.boost_user * loss_t)
            else:
                pass
            theta_tp1 = (r_t * theta_t) - (z_t * grad_theta_t)
            return theta_tp1, y_hat, loss_t

        [seq_thetas, seq_y_hats, seq_losses], _ = theano.scan(fn=recurrence, sequences=[X,Y,O,F,S,S_M1], outputs_info=[theta_0, None, None])
        seq_loss = T.mean(seq_losses)
        reg_loss = 0.0
        for reg_param in self.reg_params:
            reg_loss += T.sum(T.sqr(reg_param))
        total_loss = seq_loss #+ (self.l * reg_loss)
        grad_params = [T.grad(total_loss, p) for p in self.params]
        self.get_params = theano.function(inputs = [], outputs = self.params)
        self.get_seq_loss = theano.function([X, Y, O, F, S, S_M1, theta_0], outputs = seq_loss)
        self.get_total_loss = theano.function([X, Y, O, F, S, S_M1, theta_0], outputs = total_loss)
        self.get_seq_y_hats = theano.function([X, Y, O, F, S, S_M1, theta_0], outputs = seq_y_hats)
        self.get_seq_thetas = theano.function([X, Y, O, F, S, S_M1, theta_0], outputs = seq_thetas)
        self.get_seq_grad = theano.function([X, Y, O, F, S, S_M1, theta_0], outputs = grad_params)
        self.do_sgd_update = theano.function([X, Y, O, F, S, S_M1, theta_0, lr], outputs= [seq_loss, seq_thetas, seq_y_hats], updates = rmsprop(total_loss, self.params, lr))
        self.do_rms_update = theano.function([X, Y, O, F, S, S_M1, theta_0, lr], outputs= [seq_loss, seq_thetas, seq_y_hats], updates = sgd(total_loss, self.params, lr))
