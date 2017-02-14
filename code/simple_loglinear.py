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

def norm_init(n_in, n_out, scale=0.01, ortho=True):
    """
    Initialize weights from a scaled standard normal distribution
    Falls back to orthogonal weights if n_in = n_out
    n_in : The input dimension
    n_out : The output dimension
    scale : Scale for the normal distribution
    ortho : Fall back to ortho weights when n_in = n_out
    """
    if n_in == n_out and ortho:
        return ortho_weight(n_in)
    else:
        return floatX(scale * np.random.randn(n_in, n_out))

def ortho_weight(ndim):
    """
    Returns an orthogonal matrix via SVD decomp
    Used for initializing weight matrices of an LSTM
    """
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)


class SimpleLoglinear(object):
    def __init__(self, dh, reg = 0.1, learner_reg = 0.5, learning_model = "m1", clip = False, interpolate_bin_loss = 0, use_sum_loss = 0):
        self.dh = dh #DataHelper(event2feats_file, feat2id_file, actions_file)
        self.learning_model = learning_model
        self.clip = clip
        self.l = reg #reg parameter
        self.interpolate_bin_loss = interpolate_bin_loss
        self.use_sum_loss = use_sum_loss
        assert 0< self.interpolate_bin_loss < 1 
        assert self.use_sum_loss == 0 or self.use_sum_loss == 1
        self.ul = learner_reg #reg parameter for the learner model
        self._eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self._mult_eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
        if self.learning_model == "m1":
            x = np.ones((self.dh.FEAT_SIZE,))
            self.b_z = theano.shared(floatX(x), name='b_z')
            self.b_r = theano.shared(floatX(x), name='b_r')
            self.W_r = theano.shared(floatX(x), name='W_r')
            self.W_z = theano.shared(floatX(x), name='W_z')
            self.params = [self.W_r, self.b_r, self.W_z, self.b_z]
            self.reg_params = [self.W_r, self.W_z]
        elif self.learning_model == "m2":
            b_x = np.zeros((self.dh.FEAT_SIZE,))
            self.b_z = theano.shared(floatX(b_x), name='b_z')
            self.b_r = theano.shared(floatX(b_x), name='b_r')
            self.W_r = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, 9)), name='W_r') #9 is the size of the s_t context vector
            self.W_z = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, 9)), name='W_z') #9 is the size of the s_t context vector
            self.params = [self.W_r, self.b_r, self.W_z, self.b_z]
            self.reg_params = [self.W_r, self.W_z]
        elif self.learning_model == "m0":
            x = 1.0
            self.W_r = theano.shared(floatX(x), name='W_r')
            self.W_z = theano.shared(floatX(x), name='W_z')
            self.params = [self.W_r, self.W_z]
            self.reg_params = [self.W_r, self.W_z]
        else: 
            raise BaseException("unknown learning model")
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
        YT = T.ivector('YT') #(sequence_size,) #index of the input string
        S = T.fmatrix('S') #(sequence_size,9) # was the answer marked as correct or incorrect?
        SM1 = T.fmatrix('SM1') #(sequence_size,9) # was the answer marked as correct or incorrect?
        theta_0 = T.fvector('theta_0') #(feature_size,)

        def rms(v):
            return T.sqrt(v + self._eps)

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

        def assign_losses(loss_t, bin_loss_t, s_t):
            r_loss_t = T.switch(s_t[3], loss_t, 0)
            bin_loss_t = T.switch(s_t[3], 0, bin_loss_t)
            c_loss_t = T.switch(T.any(s_t[[4,7]]), loss_t, 0)
            ic_loss_t = T.switch(T.any(s_t[[5,8]]), loss_t, 0)
            return r_loss_t, c_loss_t, ic_loss_t, bin_loss_t

        def log_linear_t(Phi_x_t, y_t, yt_t, o_t, s_t, theta_t):
            y_dot = Phi_x_t.dot(theta_t.T) #(Y,D,) dot (D,)
            y_dot_masked = masked(y_dot, o_t, -100000000.0000) #(Y,)
            y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(Y,)
            y_hat = T.clip(y_hat_unsafe, self._eps, 0.9999999)
            model_loss = -T.sum(y_t * T.log(y_hat)) 
            m_t = T.argmax(y_hat) #model's prediction
            u_t = T.argmax(y_t)
            if m_t == u_t == yt_t:
                # model and user are correct, y_hat(yt_t) is highest so has least loss
                model_bin_loss = -T.log(y_hat[0, yt_t])
            elif m_t != yt_t and u_t == yt_t:
                #model is wrong it put too low prob on yt_t so loss is high
                model_bin_loss = -T.log(y_hat[0, yt_t])
            elif m_t == yt_t and u_t != yt_t:
                # model put too much weight on yt_t, it should have put less on it and more everywhere else
                model_bin_loss = -T.log(1.0 - y_hat[0, yt_t])
            else:
                #both model and user is wrong, model put low mass on yt_t so gets less loss
                model_bin_loss = -T.log(1.0 - y_hat[0, yt_t])

            y_target = create_target(y_t, y_hat, s_t)
            #user_loss = -T.sum(y_target * T.log(y_hat), axis=1) + self.ul * T.sum(T.sqr(theta_t))
            #theta_t_grad = T.grad(user_loss, theta_t)
            theta_t_grad = y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t) 
            if self.ul >= 0.0:
                theta_t_grad = theta_t_grad - (self.ul * 2.0 * theta_t) #obs - exp
            else:
                norm2 = T.sqrt(T.sum(T.sqr(self._eps + theta_t_grad)))
                theta_t_grad = theta_t_grad / norm2

            theta_t_grad = T.reshape(theta_t_grad, theta_t.shape) #(D,)
            y_hat = T.reshape(y_hat, (self.dh.E_SIZE,)) #(Y,)
            return theta_t_grad, y_hat, model_loss, model_bin_loss

        def recurrence(x_t, y_t, yt_t, o_t, s_t, s_tm1, theta_t):
            #x_t (scalar)
            #y_t (Y,)
            #o_t (Y,)
            #s_t (9,)
            #theta_t (D,)
            s_t = T.reshape(s_t, (9,))
            s_tm1 = T.reshape(s_tm1, (9,))
            theta_t = T.reshape(theta_t, (self.dh.FEAT_SIZE,))
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            grad_theta_t, y_hat, loss_t, bin_loss_t = log_linear_t(Phi_x_t, y_t, yt_t, o_t, s_t, theta_t) #(D,) and scalar
            r_loss_t, c_loss_t, ic_loss_t, bin_loss_t = assign_losses(loss_t, bin_loss_t, s_t)
            if s_t[6] == 1: #nofeeback no knowledge change...
                theta_tp1 = theta_t
            else:
                if self.learning_model == "m1":
                    theta_tp1 = T.nnet.sigmoid(self.W_r + self.b_r) * theta_t + T.nnet.sigmoid(self.W_z + self.b_z) * grad_theta_t
                elif self.learning_model == "m0":
                    theta_tp1 = T.nnet.sigmoid(self.W_r) * theta_t + T.nnet.sigmoid(self.W_z) * grad_theta_t
                elif self.learning_model == "m2":
                    g_r = T.nnet.sigmoid(self.W_r.dot(s_tm1)  + self.b_r)  
                    g_z = T.nnet.sigmoid(self.W_z.dot(s_t) + self.b_z)  
                    theta_tp1 = g_r * theta_t + g_z * grad_theta_t
                else:
                    raise BaseException("unknown learning model")

            if self.clip:
                theta_tp1 = T.clip(theta_tp1, -1.0, 1.0)
            else:
                pass
            return theta_tp1, y_hat, loss_t, r_loss_t,c_loss_t, ic_loss_t, bin_loss_t

        [seq_thetas, seq_y_hats, all_losses, r_losses, c_losses, ic_losses, bin_losses], _ = theano.scan(fn=recurrence, 
                sequences=[X,Y,YT,O,S,SM1], 
                outputs_info=[theta_0, None, None, None, None, None, None])

        all_loss = T.sum(all_losses)
        #r_loss_mean = T.mean(T.nonzero_values(r_losses))
        c_loss_mean = T.mean(T.nonzero_values(c_losses))
        ic_loss_mean = T.mean(T.nonzero_values(ic_losses))
        bin_loss_mean = T.mean(T.nonzero_values(bin_losses))
        mean_loss = ((1.0 - self.interpolate_bin_loss) * (c_loss_mean + ic_loss_mean)) + (self.interpolate_bin_loss * bin_loss_mean)
        #r_loss = T.sum(r_losses)
        c_loss = T.sum(c_losses)
        ic_loss = T.sum(ic_losses)
        bin_loss = T.sum(bin_losses)
        sum_loss = ((1.0 - self.interpolate_bin_loss) * (c_loss + ic_loss)) + (self.interpolate_bin_loss * bin_loss)
        reg_loss = 0.0
        for reg_param in self.reg_params:
            reg_loss += T.sum(T.abs_(reg_param + self._eps))
            reg_loss += T.sum(T.sqr(reg_param))
        model_loss = (self.use_sum_loss * sum_loss) + ((1.0 - self.use_sum_loss) * mean_loss)  
        total_loss = model_loss + (self.l * reg_loss)
        self.get_params = theano.function(inputs = [], outputs = self.params)
        self.get_seq_losses = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = [all_losses, c_losses, ic_losses, bin_losses])
        self.get_loss = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = [total_loss, model_loss, all_loss, c_loss, ic_loss, bin_loss])
        self.get_seq_y_hats = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_y_hats)
        self.get_seq_thetas = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_thetas)
        self.do_sgd_update = theano.function([X, Y, YT, O, S, SM1, theta_0, lr], 
                outputs= [total_loss, seq_thetas, seq_y_hats], 
                updates = momentum(total_loss, self.params, lr))
        self.do_rms_update = theano.function([X, Y, YT, O, S, SM1, theta_0, lr], 
                outputs= [total_loss, seq_thetas, seq_y_hats], 
                updates = sgd(total_loss, self.params, lr))
