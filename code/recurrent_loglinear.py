#!/usr/bin/env python
import numpy as np
import json
from optimizers import rmsprop
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

class RecurrentLoglinear(object):
    def __init__(self, dh, u = "rms", reg = 0.1, grad_transform= "0", grad_model = "g0", learning_model = "m1", clip = False, interpolate_bin_loss = 0, temp_model = "t0", grad_top_k = "top_all", saved_weights = None):
        self.dh = dh #DataHelper(event2feats_file, feat2id_file, actions_file)
        self.learning_model = learning_model
        self.grad_model = grad_model
        self.low_rank = 100
        self.context_size = 10
        self.grad_top_k = grad_top_k
        self.clip = clip
        self._update = rmsprop
        self.l = reg #reg parameter
        self.interpolate_bin_loss = interpolate_bin_loss
        self.use_sum_loss = 1
        self.temp_model = temp_model
        self.temp = 1.0
        self.merge = 1.0
        assert 0 <= self.interpolate_bin_loss <= 1 
        assert self.use_sum_loss == 0 or self.use_sum_loss == 1
        self.grad_transform = grad_transform 
        self._eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self._mult_eps = np.finfo(np.float32).eps #1e-10 # for fixing divide by 0
        self.phi = theano.shared(floatX(self.load_phi()), name='Phi') #(output_dim, feat_size)
        if self.learning_model == "m0":
            #scalar retention and update gate
            if saved_weights is None:
                #print 'init random weight'
                x = 0.0 #0.1  * np.random.rand(1,)
                W_r = theano.shared(floatX(x), name='W_r')
                W_z = theano.shared(floatX(x), name='W_z')
            else:
                _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
                #print '_params in load', _params
                W_r = theano.shared(floatX(_params[0]), name='W_r')
                W_z = theano.shared(floatX(_params[1]), name='W_z')
            self.params = [W_r, W_z] 
            self.reg_params = [W_r, W_z]

        elif self.learning_model == "m1":
            #vector retention and update gate
            #x = 0.1 * np.random.rand(self.dh.FEAT_SIZE,)
            if saved_weights is None:
                #print 'init random weight'
                x = 0.01 * np.random.rand(self.dh.FEAT_SIZE,) 
                b_x = 0.0 #0.01 * np.random.rand(1,)
                b_z = theano.shared(floatX(b_x), name='b_z')
                b_r = theano.shared(floatX(b_x), name='b_r')
                W_r = theano.shared(floatX(x), name='W_r')
                W_z = theano.shared(floatX(x), name='W_z')
            else:
                _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
                #print '_params in load', _params
                W_r = theano.shared(_params[0], name='W_r')
                W_z = theano.shared(_params[1], name='W_z')
                b_z = theano.shared(_params[2], name='b_z')
                b_r = theano.shared(_params[3], name='b_r')
            self.params = [W_r, W_z, b_z, b_r]
            self.reg_params = [W_r, W_z]
        elif self.learning_model == "m2":
            raise BaseException("model option removed.. see extra model file")
        elif self.learning_model == "m3":
            # vector retention and update with context   
            if saved_weights is None:
                #print 'init random weight'
                _b_x = 0.0
                W = np.random.randn(self.context_size, self.context_size)
                u, s, v = np.linalg.svd(W)
                b_z = theano.shared(floatX(_b_x), name="b_z") 
                b_r = theano.shared(floatX(_b_x), name="b_r")
                W_rc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_rc') 
                W_zc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name='W_zc') 
                W_rc2 = theano.shared(floatX(u), name='W_rc1')
                W_zc2 = theano.shared(floatX(u), name='W_zc2')
            else:
                _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
                #print '_params in load', _params
                W_zc = theano.shared(floatX(_params[0]), name='W_zc') 
                W_rc = theano.shared(floatX(_params[1]), name='W_rc') 
                W_zc2 = theano.shared(floatX(_params[2]), name='W_zc2')
                W_rc2 = theano.shared(floatX(_params[3]), name='W_rc1')
                b_z = theano.shared(floatX(_params[4]), name="b_z") 
                b_r = theano.shared(floatX(_params[5]), name="b_r")
            self.params = [W_zc, W_rc, W_zc2, W_rc2, b_z, b_r]
            self.reg_params = [W_zc, W_rc, W_zc2, W_rc2]
        elif self.learning_model == "m3.3":
            raise BaseException("model option removed.. see extra model file")
        elif self.learning_model == "m4":
            if saved_weights is None:
                #print 'init random weight'
                _b_x = 0.0
                b_z = theano.shared(floatX(_b_x), name="b_z") 
                b_r = theano.shared(floatX(_b_x), name="b_r")
                W_rc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, 5 * self.context_size)), name='W_rc') 
                W_zc = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, 5 * self.context_size)), name='W_zc') 
                W_rc2 = theano.shared(floatX(0.01 * np.random.rand(self.context_size * 5, self.context_size + (3 * self.dh.E_SIZE))), name='W_rc2') 
                W_zc2 = theano.shared(floatX(0.01 * np.random.rand(self.context_size * 5, self.context_size + (3 * self.dh.E_SIZE))), name='W_zc2') 
            else:
                _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
                #print '_params in load', _params
                W_zc = theano.shared(floatX(_params[0]), name='W_zc') 
                W_rc = theano.shared(floatX(_params[1]), name='W_rc') 
                W_zc2 = theano.shared(floatX(_params[2]), name='W_zc2')
                W_rc2 = theano.shared(floatX(_params[3]), name='W_rc2')
                b_z = theano.shared(floatX(_params[4]), name="b_z") 
                b_r = theano.shared(floatX(_params[5]), name="b_r")
            self.params = [W_zc, W_rc, W_zc2, W_rc2, b_z, b_r]
            self.reg_params = [W_zc, W_rc, W_zc2, W_rc2]
        elif self.learning_model == "m5":
            raise BaseException("model option removed.. see extra model file")
        elif self.learning_model == "m6":
            raise BaseException("model option removed.. see extra model file")
        else: 
            raise BaseException("unknown learning model")

        if self.temp_model == "t0":
            self.b_temp = theano.shared(floatX(1.0), name='b_temp')
        elif self.temp_model == "t1":
            raise BaseException("temp option removed.. see extra model file")
        elif self.temp_model == "t2":
            raise BaseException("temp option removed.. see extra model file")
        else:
            raise BaseException("unknown temp model")

        if self.grad_model == "g2":
            if saved_weights is None:
                W = np.random.randn(self.context_size, self.context_size)
                u, s, v = np.linalg.svd(W)
                b_m = theano.shared(floatX(1.0), name = 'b_m')
                W_m1 = theano.shared(floatX(0.01 * np.random.rand(self.dh.FEAT_SIZE, self.context_size)), name= "W_m1")
                W_m2 = theano.shared(floatX(u), name= "W_m2")
            else:
                _params = [floatX(np.asarray(i)) for i in json.loads(open(saved_weights, 'r').read())]
                #print '_params in load in g2', _params
                b_m = theano.shared(floatX(_params[-1]), name = 'b_m')
                W_m2 = theano.shared(floatX(_params[-2]), name= "W_m2")
                W_m1 = theano.shared(floatX(_params[-3]), name= "W_m1")
            self.params += [W_m1, W_m2, b_m]
            self.reg_params += [W_m1, W_m2]
        elif self.grad_model == "g1":
            self.b_m = theano.shared(floatX(1.0), name = 'b_m')
        elif self.grad_model == "g0":
            self.b_m = theano.shared(floatX(1.0), name = 'b_m')
        elif self.grad_model == "g3":
            self.b_m = theano.shared(floatX(1.0), name = 'b_m')
        else:
            raise BaseException("unknown grad model")
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

    def save_weights(self, save_path):
        #_tmp_params = self.get_params()
        #print '_params in save', _tmp_params
        _params = json.dumps([i.get_value().tolist() for i in self.params])
        f = open(save_path, 'w')
        f.write(_params)
        f.flush()
        f.close()
        return _params

    def make_graph(self):
        lr = T.scalar('lr', dtype=theano.config.floatX) # learning rate scalar
        X = T.ivector('X') #(sequence_size,) #index of the input string
        O = T.fmatrix('O') #(sequence_size, output_dim) #mask for possible Ys
        Y = T.fmatrix('Y') #(sequence_size, output_dim) #the Y that is selected by the user
        YT = T.ivector('YT') #(sequence_size,) #index of the input string
        S = T.fmatrix('S') #(sequence_size,self.context_size) # was the answer marked as correct or incorrect?
        SM1 = T.fmatrix('SM1') #(sequence_size,self.context_size) # was the answer marked as correct or incorrect?
        theta_0 = T.fvector('theta_0') #(feature_size,)
        _x_t = T.iscalar('_x_t')
        _o_t = T.fvector('_o_t')
        _theta_tm1 = T.fvector('_theta_tm1')

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

        def new_softmax(v):
            s_v = T.reshape(v, (1, self.dh.E_SIZE))
            e_v = T.exp(s_v / self.temp) #TEMP is fixed at 1
            return e_v / T.sum(e_v)

        def assign_losses(loss_t, bin_loss_t, c_t):
            r_loss_t = T.switch(c_t[3], loss_t, 0)
            bin_loss_t = T.switch(c_t[3], 0, bin_loss_t)
            c_loss_t = T.switch(T.any(c_t[[4,7]]), loss_t, 0)
            ic_loss_t = T.switch(T.any(c_t[[5,8]]), loss_t, 0)
            return r_loss_t, c_loss_t, ic_loss_t, bin_loss_t

        def obs_model(x_t, o_t, theta_tm1):
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            y_dot = Phi_x_t.dot(theta_tm1.T) #(Y,D,) dot (D,)
            y_dot_masked = masked(y_dot, o_t, -1e8) #(1,Y)
            #y_hat_unsafe  = T.nnet.softmax(y_dot_masked) #(1,Y)
            y_hat_unsafe  = new_softmax(y_dot_masked) #T.nnet.softmax(y_dot_masked) #(1,Y)
            y_hat = T.clip(y_hat_unsafe, floatX(self._eps), floatX(1.0 - self._eps))
            return y_hat, Phi_x_t

        def compute_losses(y_hat, y_t, yt_t):
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
            return model_loss, model_bin_loss

        def compute_update(x_t, y_hat, y_t, c_t):
            Phi_x_t = self.phi[x_t, :, :] #(1, Y, D)
            Phi_x_t = T.reshape(Phi_x_t, (self.dh.E_SIZE, self.dh.FEAT_SIZE)) #(Y,D)
            if self.grad_model == "g0":
                #Redistribution update scheme
                y_target = create_target(y_t, y_hat, c_t)
                theta_t_grad = y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t) 
            elif self.grad_model == "g1":
                #Negative update scheme
                pos_theta_t_grad = y_t.dot(Phi_x_t) - y_hat.dot(Phi_x_t)
                theta_t_grad = T.switch(T.eq(c_t[5], 1.0), -pos_theta_t_grad, pos_theta_t_grad)
                #if c_t[5] == 1: #if c_t[5] is 1 then the student knows their answer is wrong... so we use the "reverse" gradient
                #    theta_t_grad = -theta_t_grad
                #else:
                #    pass
            elif self.grad_model == "g2":
                #Interpolated REDISTRIBUTION AND NEGATIVE update scheme
                #pos_theta_t_grad = y_t.dot(Phi_x_t) - y_hat.dot(Phi_x_t)
                #neg_theta_t_grad = -pos_theta_t_grad
                #y_target = create_target(y_t, y_hat, c_t)
                #redistribute_theta_t_grad = y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t) 
                #merge_theta_t_grad = merge * neg_theta_t_grad + (1.0 - merge) * redistribute_theta_t_grad
                #theta_t_grad = T.switch(T.eq(c_t[5],1.0), merge_theta_t_grad, pos_theta_t_grad)
                #if c_t[5] == 1:
                #    y_target = create_target(y_t, y_hat, c_t)
                #    theta_t_grad_g2 = y_target.dot(Phi_x_t) - y_hat.dot(Phi_x_t) 
                #    theta_t_grad_g1 = -theta_t_grad
                #    theta_t_grad = merge * theta_t_grad_g1 + (1.0 - merge) * theta_t_grad_g2
                #else:
                #    theta_t_grad = merge * theta_t_grad_c + (1.0 - merge) * theta_t_grad_c
                pass
            elif self.grad_model == "g3":
                #Feature Vector Update scheme
                y_t_idx = T.argmax(y_t)
                Phi_x_y = Phi_x_t[y_t_idx,:] #(D,)
                Phi_x_y = T.reshape(Phi_x_y, (self.dh.FEAT_SIZE,)) #(Y,D)
                theta_t_grad = T.switch(T.eq(c_t[5],1.0), Phi_x_y, -Phi_x_y)
                norm = T.sum(theta_t_grad)
                theta_t_grad = theta_t_grad / norm
            else:
                raise BaseException("unknown user grad type")

            if self.grad_transform == "0":
                theta_t_grad = theta_t_grad #- (self.grad_transform * 2.0 * theta_t) #obs - exp
            elif self.grad_transform == "norm":
                norm2 = T.sqrt(T.sum(T.sqr(self._eps + theta_t_grad)))
                theta_t_grad = theta_t_grad / norm2
            elif self.grad_transform == "sign":
                theta_t_grad = T.switch(T.lt(theta_t_grad, 0), floatX(-0.01), theta_t_grad)
                theta_t_grad = T.switch(T.gt(theta_t_grad, 0), floatX(0.01), theta_t_grad)
            else:
                raise BaseException("unknown user ul")
            theta_t_grad = T.reshape(theta_t_grad, (self.dh.FEAT_SIZE,)) #(D,)
            if self.grad_top_k == "top_all":
                pass
            elif self.grad_top_k.startswith("top_"):
                k = int(self.grad_top_k.split("_")[1])
                theta_t_grad_abs = T.abs_(theta_t_grad)
                theta_t_grad_abs_argsort = T.argsort(theta_t_grad_abs)
                theta_t_grad = T.set_subtensor(theta_t_grad[theta_t_grad_abs_argsort[:self.dh.FEAT_SIZE - k]],0)
            else:
                raise BaseException("unknown grad top k")
            return theta_t_grad

        def log_linear_t(x_t, y_t, yt_t, o_t, c_t, theta_tm1):
            y_hat, Phi_x_t = obs_model(x_t, o_t, theta_tm1)
            model_loss, model_bin_loss = compute_losses(y_hat, y_t, yt_t)
            theta_t_grad = compute_update(x_t, y_hat, y_t, c_t)
            y_hat = T.reshape(y_hat, (self.dh.E_SIZE,)) #(Y,)
            return theta_t_grad, y_hat, model_loss, model_bin_loss

        def transition_model(x_t, y_t, yt_t, o_t, s_t, s_tm1, theta_tm1):
            c_t = s_t[:10] #T.set_subtensor(s_t[[6,7,8]],0)
            c_tm1 = s_tm1[:10] #T.set_subtensor(s_tm1[[6, 7,8]],0)
            c_t = T.reshape(c_t, (self.context_size,))
            c_tm1 = T.reshape(c_tm1, (self.context_size,))
            #update_t, y_hat, loss_t, bin_loss_t = log_linear_t(x_t, y_t, yt_t, o_t, c_t, merge, temp, theta_tm1) #(D,) and scalar
            y_hat, Phi_x_t = obs_model(x_t, o_t, theta_tm1)
            update_t = compute_update(x_t, y_hat, y_t, c_t)
            if self.learning_model == "m0":
                W_r = self.params[0]
                W_z = self.params[1]
                g_r = T.nnet.sigmoid(W_r)  
                g_z = T.nnet.sigmoid(W_z)  
                theta_t = g_r * theta_tm1 + g_z * update_t
            elif self.learning_model == "m1":
                W_r = self.params[0]
                W_z = self.params[1]
                b_z = self.params[2]
                b_r = self.params[3]
                g_r = T.nnet.sigmoid(W_r + b_r)
                g_z = T.nnet.sigmoid(W_z + b_z) 
                theta_t = g_r * theta_tm1 + g_z * update_t
            elif self.learning_model == "m3":
                W_zc = self.params[0]
                W_rc = self.params[1]
                W_zc2 = self.params[2]
                W_rc2 = self.params[3]
                b_z = self.params[4]
                b_r = self.params[5]
                g_r = T.nnet.sigmoid(W_rc.dot(W_rc2.dot(c_tm1)) + b_r)
                g_z = T.nnet.sigmoid(W_zc.dot(W_zc2.dot(c_t)) + b_z) #<--- everything but input x
                theta_t = g_r * theta_tm1 + g_z * update_t
            elif self.learning_model == 'm4':
                W_zc = self.params[0]
                W_rc = self.params[1]
                W_zc2 = self.params[2]
                W_rc2 = self.params[3]
                b_z = self.params[4] 
                b_r = self.params[5]
                g_r = T.nnet.sigmoid(W_rc.dot(W_rc2.dot(s_tm1)) + b_r)
                g_z = T.nnet.sigmoid(W_zc.dot(W_zc2.dot(s_t)) + b_z) #<--- everything but input x
                theta_t = g_r * theta_tm1 + g_z * update_t
            else:
                raise BaseException("unknown learning model")
            theta_t = T.switch(T.eq(c_t[6], 1.0), theta_tm1, theta_t) #if c_t has no feedback then do not change theta...
            if self.clip:
                theta_t = T.clip(theta_t, -1.0, 1.0)
            else:
                pass
            return theta_t, g_r, g_z
        
        def recurrence(x_t, y_t, yt_t, o_t, s_t, s_tm1, theta_tm1):
            #x_t (scalar)
            #y_t (Y,)
            #o_t (Y,)
            #s_t (self.context_size,)
            #theta_tm1 (D,)
            s_t = T.reshape(s_t, (self.context_size + (3  * self.dh.E_SIZE),))
            s_tm1 = T.reshape(s_tm1, (self.context_size + (3  * self.dh.E_SIZE),))
            c_t = s_t[:10] #T.set_subtensor(s_t[[6,7,8]],0)
            c_tm1 = s_tm1[:10] #T.set_subtensor(s_tm1[[6, 7,8]],0)
            c_t = T.reshape(c_t, (self.context_size,))
            c_tm1 = T.reshape(c_tm1, (self.context_size,))
            theta_tm1 = T.reshape(theta_tm1, (self.dh.FEAT_SIZE,))
            update_t, y_hat, loss_t, bin_loss_t = log_linear_t(x_t, y_t, yt_t, o_t, c_t, theta_tm1) #(D,) and scalar
            theta_t, g_r, g_z = transition_model(x_t, y_t, yt_t, o_t, s_t, s_tm1, theta_tm1)
            r_loss_t, c_loss_t, ic_loss_t, bin_loss_t = assign_losses(loss_t, bin_loss_t, c_t)
            return theta_t, y_hat, loss_t, r_loss_t, c_loss_t, ic_loss_t, bin_loss_t, update_t, g_r, g_z

        [seq_thetas, seq_y_hats, all_losses, r_losses, c_losses, ic_losses, bin_losses, seq_updates, seq_g_r, seq_g_z], _ = theano.scan(fn=recurrence, 
                sequences=[X,Y,YT,O,S,SM1], 
                outputs_info=[theta_0, None, None, None, None, None, None, None, None, None])

        all_loss = T.sum(all_losses)
        #def log_linear_t(x_t, y_t, yt_t, o_t, c_t, merge, temp, theta_tm1):

        _y_hat, _phi_x_t = obs_model(_x_t, _o_t, _theta_tm1)
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
        self.get_step_y_hat = theano.function(inputs=[_x_t, _o_t, _theta_tm1], outputs=[_y_hat])
        self.get_params = theano.function(inputs = [], outputs = [T.as_tensor_variable(p) for p in self.params])
        self.get_seq_losses = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = [all_losses, c_losses, ic_losses, bin_losses])
        self.get_loss = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = [total_loss, model_loss, all_loss, c_loss, ic_loss, bin_loss])
        self.get_seq_y_hats = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_y_hats)
        self.get_seq_thetas = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_thetas)
        self.get_seq_updates = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_updates)
        self.get_seq_g_r = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_g_r)
        self.get_seq_g_z = theano.function([X, Y, YT, O, S, SM1, theta_0], outputs = seq_g_z)
        self.do_update = theano.function([X, Y, YT, O, S, SM1, theta_0, lr], 
                outputs= [total_loss, seq_thetas, seq_y_hats], 
                updates = self._update(total_loss, self.params, lr))
