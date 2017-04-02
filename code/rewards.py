#!/usr/bin/env python
import numpy as np
import theano
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

#from usermodel import UserModel

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64

np.set_printoptions(precision=4)
np.seterr(all = 'raise')

class Reward(object):
    def __init__(self, datahelper):
        self.dh = datahelper

    def get_ll_reward(self, rll, theta_t):
        r = 0.0
        for qa in self.dh.quiz_action_vectors:
            a_type, x_t, yt_t, o_t = qa
            y_hat = rll.get_step_y_hat(x_t, o_t, theta_t)
            y_hat = np.reshape(y_hat, (self.dh.E_SIZE,))
            r += y_hat[yt_t]
        return r
