#!/usr/bin/env python
import numpy as np
import theano
import sys
import codecs
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
import pdb

#from usermodel import UserModel

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64

np.set_printoptions(precision=4)
np.seterr(all = 'raise')

class Reward(object):
    def __init__(self, datahelper, reward_type='ll'):
        self.dh = datahelper
        self.reward_type = reward_type

    def get_reward(self, rll, theta_t):
        if self.reward_type == 'll':
            return self.get_ll_reward(rll, theta_t)
        elif self.reward_type == 'match':
            return self.get_match_reward(rll, theta_t)
        else:
            raise Exception("unknown reward type")
        return True

    def get_match_reward(self, rll, theta_t):
        r = 0.0
        for qa in self.dh.quiz_action_vectors:
            a_type, x_t, yt_t, o_t = qa
            y_hat = rll.get_step_y_hat(x_t, o_t, theta_t)
            y_hat = np.reshape(y_hat, (self.dh.E_SIZE,))
            y_max_idx = np.argmax(y_hat)
            phi_max = self.dh._phi(x_t, y_max_idx)
            phi_true = self.dh._phi(x_t, yt_t)
            feature_overlap = phi_max * phi_true 
            feature_overlap_ratio = np.sum(feature_overlap) / np.sum(phi_true)
            r += feature_overlap_ratio
        return r / float(len(self.dh.quiz_action_vectors))

    def get_ll_reward(self, rll, theta_t):
        r = 0.0
        for qa in self.dh.quiz_action_vectors:
            a_type, x_t, yt_t, o_t = qa
            y_hat = rll.get_step_y_hat(x_t, o_t, theta_t)
            y_hat = np.reshape(y_hat, (self.dh.E_SIZE,))
            r += y_hat[yt_t]
        return r
