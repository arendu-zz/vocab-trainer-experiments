#!/usr/bin/env python
import numpy as np
from optimizers import sgd, rmsprop, momentum
import theano
import theano.tensor as T

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    intX = np.int32
    floatX = np.float32
else:
    intX = np.int64
    floatX = np.float64
"""
def norm_init(n_in, n_out, scale=0.01, ortho=True):
    if n_in == n_out and ortho:
        return ortho_weight(n_in)
    else:
        return floatX(scale * np.random.randn(n_in, n_out))

def ortho_weight(ndim):
    W = np.random.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype(theano.config.floatX)
"""

class SimpleLoglinear(object):
    def __init__(self, dh, u = "sgd", reg = 0.1, learner_reg = 0.5, grad_model = "g0", learning_model = "m1", clip = False, interpolate_bin_loss = 0, temp_model = "t0"):
        self.dh = dh #DataHelper(event2feats_file, feat2id_file, actions_file)
        self.learning_model = learning_model
        self.grad_model = grad_model
        self.low_rank = 100
        self.context_size = 10
        self.clip = clip
        self._update = None
        if u == "sgd":
            self._update = sgd
        elif u == "rms":
            self._update = rmsprop
        elif u == 'mom':
            self._update = momentum
        else:
            raise Exception("unknown grad update:" + u)
        self.l = reg #reg parameter
        self.interpolate_bin_loss = interpolate_bin_loss
        self.use_sum_loss = 1
        self.temp_model = temp_model
        assert 0 <= self.interpolate_bin_loss <= 1 
        assert self.use_sum_loss == 0 or self.use_sum_loss == 1
        self.ul = learner_reg #reg parameter for the learner model
        self._eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self._mult_eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
        if self.learning_model == "m1":
            #x = 0.1 * np.random.rand(self.dh.FEAT_SIZE,)
            x = 0.01 * np.random.rand(self.dh.FEAT_SIZE,) 
            b_x = 0.0 #0.01 * np.random.rand(1,)
            self.b_z = theano.shared(floatX(b_x), name='b_z')
            self.b_r = theano.shared(floatX(b_x), name='b_r')
            self.W_r = theano.shared(floatX(x), name='W_r')
            self.W_z = theano.shared(floatX(x), name='W_z')
            self.params = [self.W_r, self.b_r, self.W_z, self.b_z]
            self.reg_params = [self.W_r, self.W_z]
        elif self.learning_model == "m2":
            #x = 0.1 * np.random.rand(self.dh.FEAT_SIZE, self.context_size) #self.context_size is the size of the s_t context vector 
            x = 0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size) #self.context_size is the size of the s_t context vector 
            b_x = 0.0 #0.01 * np.random.rand(1,))
            self.b_z = theano.shared(floatX(b_x), name='b_z')
            self.b_r = theano.shared(floatX(b_x), name='b_r')
            self.W_rc = theano.shared(floatX(x), name='W_rc') 
            self.W_zc = theano.shared(floatX(x), name='W_zc') 
            self.params = [self.W_rc, self.b_r, self.W_zc, self.b_z]
            self.reg_params = [self.W_rc, self.W_zc]
        elif self.learning_model == "m0":
            x = 0.0 #0.1  * np.random.rand(1,)
            self.W_r = theano.shared(floatX(x), name='W_r')
            self.W_z = theano.shared(floatX(x), name='W_z')
            self.params = [self.W_r, self.W_z] #self.context_size is the size of the s_t context vector
            self.reg_params = [self.W_r, self.W_z]
        elif self.learning_model == "m3":
            _b_x = 0.0
            self.b_z = theano.shared(floatX(_b_x), name="b_z") 
            self.b_r = theano.shared(floatX(_b_x), name="b_r")
            self.W_rc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_rc') 
            self.W_zc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_zc') 
            self.W_rx = theano.shared(floatX(0.01 * np.random.rand(self.dh.E_SIZE,)), name='W_rx')
            self.W_zx = theano.shared(floatX(0.01 * np.random.rand(self.dh.E_SIZE,)), name='W_zx')
            self.params = [self.W_zc, self.W_rc, self.W_zx, self.W_rx, self.b_r, self.b_z]
            self.reg_params = [self.W_zc, self.W_rc, self.W_zx, self.W_rx]
        elif self.learning_model == "m4":
            self.W_rt = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_rt')
            self.b_r_t = theano.shared(floatX(0.01 * np.random.rand(self.low_rank,)), name="b_r_t")

            self.W_zt = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_zt')
            self.b_z_t = theano.shared(floatX(0.01 * np.random.rand(self.low_rank,)), name="b_z_t")

            self.W_rm = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank + self.context_size)), name='W_rm') 
            self.W_zm = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank + self.context_size)), name='W_zm') 
            _b_x = 0.0
            self.b_z = theano.shared(floatX(_b_x), name="b_z") 
            self.b_r = theano.shared(floatX(_b_x), name="b_r")

            self.params = [self.W_rt, self.b_r_t, self.W_zt, self.b_z_t, self.W_rm, self.W_zm, self.b_r, self.b_z]
            self.reg_params = [self.W_rt, self.W_zt, self.W_rm, self.W_zm]
        elif self.learning_model == "m5":
            _b_x = 0.0
            self.b_z = theano.shared(floatX(_b_x), name="b_z") 
            self.b_r = theano.shared(floatX(_b_x), name="b_r")
            self.W_rt1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank)), name='W_rt1')
            self.W_rt2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_rt2')
            self.W_zg1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank)), name='W_zg1')
            self.W_zg2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_zg2')
            self.W_rc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_rc') 
            self.W_zc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_zc') 
            self.W_rx = theano.shared(floatX(0.01 * np.random.rand(self.dh.E_SIZE,)), name='W_rx')
            self.W_zx = theano.shared(floatX(0.01 * np.random.rand(self.dh.E_SIZE,)), name='W_zx')
            self.params = [self.W_rx, self.W_zx, self.W_rt1, self.W_rt2, self.W_zg1, self.W_zg2, self.W_zc, self.W_rc, self.b_r, self.b_z]
            self.reg_params = [self.W_rx, self.W_zx, self.W_rt1, self.W_rt2, self.W_zg1, self.W_zg2, self.W_zc, self.W_rc]
        elif self.learning_model == "m6":
            b_x = 0.0 
            self.W_rt1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank)), name='W_rt1')
            self.W_rt2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_rt2')
            self.W_zt1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank)), name='W_zt1')
            self.W_zt2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_zt2')

            self.W_rx = theano.shared(floatX(0.01 * np.random.rand(self.dh.E_SIZE,)), name='W_rx1')
            self.W_zx = theano.shared(floatX(0.01 * np.random.rand(self.dh.E_SIZE,)), name='W_zx1')

            self.W_rc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_rc') 
            self.W_zc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_zc') 

            self.b_z = theano.shared(floatX(b_x), name='b_z')
            self.b_r = theano.shared(floatX(b_x), name='b_r')
            self.params = [self.W_rx, self.W_rt1, self.W_rt2, self.W_zt1, self.W_zt2, self.W_zx, self.W_zc, self.W_rc, self.b_r, self.b_z] 
            self.reg_params = [self.W_rx, self.W_rt1, self.W_rt2, self.W_zt1, self.W_zt2, self.W_zx, self.W_zc, self.W_rc]
        else: 
            raise BaseException("unknown learning model")
        if self.grad_model == "g0":
            pass
        elif self.grad_model == "g1":
            self.W_g1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.low_rank)), name='W_g1')
            self.W_g2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name='W_g2')
            self.b_g1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE,)), name= 'b_g1')
            self.b_g2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank,)), name= 'b_g2')
            self.reg_params += [self.W_g1, self.W_g2]
            self.params += [self.W_g1, self.W_g2, self.b_g2, self.b_g1]
        else:
            raise BaseException("unknown grad model")

        if self.temp_model == "t0":
            self.b_temp = theano.shared(floatX(1.0), name='b_temp')
        elif self.temp_model == "t1":
            self.b_temp = theano.shared(floatX(0.0), name='b_temp')
            self.params += [self.b_temp]
        elif self.temp_model == "t2":
            self.W_temp_c = theano.shared(floatX(0.01 * np.random.rand(4,)), name="W_temp_c")
            self.b_temp = theano.shared(floatX(0.0), name="b_temp")
            self.params += [self.W_temp_c, self.b_temp]
            self.reg_params += [self.W_temp_c]
        elif self.temp_model == "t3":
            self.W_temp_c = theano.shared(floatX(0.01 * np.random.rand(4,)), name="W_temp_c")
            self.W_temp_theta1 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank, self.dh.FEAT_SIZE)), name="W_temp_theta1")
            self.W_temp_theta2 = theano.shared(floatX(0.01 * np.random.rand(self.low_rank,)), name="W_temp_theta2")
            self.b_temp = theano.shared(floatX(0.0), name="b_temp")
            self.b_temp_2 = theano.shared(floatX(np.zeros(self.low_rank,)), name="b_temp_2")
            self.params += [self.W_temp_c, self.W_temp_theta1, self.W_temp_theta2, self.b_temp, self.b_temp_2]
            self.reg_params += [self.W_temp_c, self.W_temp_theta1, self.W_temp_theta2]
        else:
            raise BaseException("unknown temp model")
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
        S = T.fmatrix('S') #(sequence_size,self.context_size) # was the answer marked as correct or incorrect?
        SM1 = T.fmatrix('SM1') #(sequence_size,self.context_size) # was the answer marked as correct or incorrect?
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

        def new_softmax(v, temp):
            s_v = T.reshape(v, (1, self.dh.E_SIZE))
            e_v = T.exp(s_v / temp)
            return e_v / T.sum(e_v)

        def assign_losses(loss_t, bin_loss_t, s_t):
            r_loss_t = T.switch(s_t[3], loss_t, 0)
            bin_loss_t = T.switch(s_t[3], 0, bin_loss_t)
            c_loss_t = T.switch(T.any(s_t[[4,7]]), loss_t, 0)
            ic_loss_t = T.switch(T.any(s_t[[5,8]]), loss_t, 0)
            return r_loss_t, c_loss_t, ic_loss_t, bin_loss_t

        def log_linear_t(Phi_x_t, y_t, yt_t, o_t, s_t, temp, theta_t):
            y_dot = Phi_x_t.dot(theta_t.T) #(Y,D,) dot (D,)
            y_dot_masked = masked(y_dot, o_t, -100000000.0000) #(1,Y)
            #y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(1,Y)
            y_hat_unsafe  = new_softmax(y_dot_masked, temp) #T.nnet.softmax(y_dot_masked) #(1,Y)
            y_hat = T.clip(y_hat_unsafe, floatX(self._eps), floatX(1.0 - self._eps))
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
            #user_loss = -T.sum(y_target * T.log(y_hat)) + self.ul * T.sum(T.sqr(theta_t))
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
            #s_t (self.context_size,)
            #theta_t (D,)
            s_t = T.reshape(s_t, (self.context_size,))
            s_tm1 = T.reshape(s_tm1, (self.context_size,))
            theta_t = T.reshape(theta_t, (self.dh.FEAT_SIZE,))
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            if self.temp_model == "t0":
                temp = self.b_temp #always 1.0
            elif self.temp_model == "t1":
                temp = 3 * T.clip(T.nnet.sigmoid(self.b_temp), 0.1, 1.0)
            elif self.temp_model == "t2":
                c_temp = self.W_temp_c.dot(T.concatenate((s_t[:3], s_t[-1:])))
                temp = 3 * T.clip(T.nnet.sigmoid(self.b_temp + c_temp), 0.1, 1.0)
            elif self.temp_model == "t3":
                c_temp = self.W_temp_c.dot(T.concatenate((s_t[:3], s_t[-1:])))
                t_temp = self.W_temp_theta2.dot(T.nnet.sigmoid(self.b_temp_2 + self.W_temp_theta1.dot(theta_t)))
                temp = 3 * T.clip(T.nnet.sigmoid(self.b_temp + c_temp + t_temp), 0.1, 1.0)
            else:
                raise BaseException("unknown temp model")
            o_gtheta_t, y_hat, loss_t, bin_loss_t = log_linear_t(Phi_x_t, y_t, yt_t, o_t, s_t, temp, theta_t) #(D,) and scalar
            r_loss_t, c_loss_t, ic_loss_t, bin_loss_t = assign_losses(loss_t, bin_loss_t, s_t)
            if s_t[6] == 1: #nofeeback no knowledge change...
                theta_tp1 = theta_t
            else:
                if self.grad_model == "g0":
                    gtheta_t = o_gtheta_t
                elif self.grad_model == "g1":
                    gtheta_t = T.tanh(self.W_g1.dot((T.tanh(self.W_g2.dot(o_gtheta_t) + self.b_g2))) + self.b_g1)
                else:
                    raise BaseException("unknown grad model")

                if self.learning_model == "m0":
                    theta_tp1 = T.nnet.sigmoid(self.W_r) * theta_t + T.nnet.sigmoid(self.W_z) * gtheta_t
                elif self.learning_model == "m1":
                    theta_tp1 = T.nnet.sigmoid(self.W_r + self.b_r) * theta_t + T.nnet.sigmoid(self.W_z + self.b_z) * gtheta_t
                elif self.learning_model == "m2":
                    g_r = T.nnet.sigmoid(self.W_rc.dot(s_tm1)  + self.b_r)  
                    g_z = T.nnet.sigmoid(self.W_zc.dot(s_t) + self.b_z)  
                    theta_tp1 = g_r * theta_t + g_z * gtheta_t
                elif self.learning_model == "m3":
                    g_r = T.nnet.sigmoid(self.W_rc.dot(s_tm1) + self.W_rx.dot(Phi_x_t) + self.b_r)
                    g_z = T.nnet.sigmoid(self.W_zc.dot(s_t) + self.W_zx.dot(Phi_x_t) + self.b_z) #<--- everything but input x
                    theta_tp1 = g_r * theta_t + g_z * gtheta_t
                elif self.learning_model == "m4":
                    lr_theta_r = T.nnet.sigmoid(self.W_rt.dot(theta_t) + self.b_r_t) #(low_rank,)
                    lr_theta_z = T.nnet.sigmoid(self.W_zt.dot(theta_t) + self.b_z_t) #(low_rank,)
                    g_r = T.nnet.sigmoid(self.W_rm.dot(T.concatenate((lr_theta_r, s_tm1))) + self.b_r) 
                    g_z = T.nnet.sigmoid(self.W_zm.dot(T.concatenate((lr_theta_z, s_t))) + self.b_z) 
                    theta_tp1 = g_r * theta_t + g_z * gtheta_t
                elif self.learning_model == "m5":
                    raise NotImplementedError("m5 not implemented")
                elif self.learning_model == "m6":
                    g_r = T.nnet.sigmoid(self.W_rt1.dot(self.W_rt2.dot(theta_t)) + self.W_rc.dot(s_tm1) + self.W_rx.dot(Phi_x_t) + self.b_r)
                    g_z = T.nnet.sigmoid(self.W_zt1.dot(self.W_zt2.dot(theta_t)) + self.W_zc.dot(s_t) + self.W_zx.dot(Phi_x_t) + self.b_z) #<----- does not use gtheta
                    theta_tp1 = g_r * theta_t + g_z * gtheta_t
                else:
                    raise BaseException("unknown learning model")

            if self.clip:
                theta_tp1 = T.clip(theta_tp1, -1.0, 1.0)
            else:
                pass
            return theta_tp1, y_hat, loss_t, r_loss_t, c_loss_t, ic_loss_t, bin_loss_t

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
        self.do_update = theano.function([X, Y, YT, O, S, SM1, theta_0, lr], 
                outputs= [total_loss, seq_thetas, seq_y_hats], 
                updates = self._update(total_loss, self.params, lr))
