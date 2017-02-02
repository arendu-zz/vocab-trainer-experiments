#!/usr/bin/env python
from code.loglinear import UserModel
import pdb
import sys
import codecs
import numpy as np
import theano
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64


if __name__ == '__main__':
    events_file = './data/content/fake-en-medium.p.w.c.event2feats'
    feats_file = './data/content/fake-en-medium.p.w.c.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    um = UserModel(events_file, feats_file, actions_file)
    print um.dh.E_SIZE
    print um.dh.F_SIZE
    print um.dh.FEAT_SIZE
    for g_id in xrange(5):
        prev_user = None
        for line in codecs.open('./data/data_splits/new-group' + str(g_id) + '.data', 'r', 'utf8').readlines():
            user, uts, ptype, tstep, a_idx, fr, en_options, en_selected, fb  = [i.strip() for i in line.split('\t')]
            if user != prev_user:
                print 'new user', user
                W = 0.001 * np.ones((um.dh.FEAT_SIZE,)).astype(floatX)
                b = 0.001 * np.ones((um.dh.E_SIZE,)).astype(floatX)
            prev_user = user

            if en_selected != "NO_ANSWER_MADE":
                x = np.array([um.dh.f2id[fr]]) #x
                e_id = um.dh.e2id[en_selected] #y
                y_selected = np.zeros((1, um.dh.E_SIZE))
                y_selected[0, e_id] = 1.0
                if en_options == "ALL":
                    o = np.ones((1, um.dh.E_SIZE)).astype(floatX)
                else:
                    o = np.zeros((1, um.dh.E_SIZE)).astype(floatX)
                    for os in en_options.split(','):
                        o_id = um.dh.e2id[os.strip()]
                        o[0, o_id] = 1.0
                f = np.array([0], dtype=bool) if fb == 'incorrect' else np.array([1], dtype=bool)
                x = x.astype(np.int64)
                o = o.astype(np.int64)
                y_selected = y_selected.astype(np.int64)
                f = f.astype(np.int64)
                #phi = um.get_phi_x(x)
                #print 'phi_shape', phi.shape
                #y_d = um.y_dot_x(W, b, x) 
                #print 'y_dot', y_d
                #y_dm = um.y_dot_masked_x(W, b, x, o) 
                #print 'y_dot_masked', y_dm
                y_h  = um.y_given_x(W, b, x, o)
                #print 'y_given_x', y_h
                #print 'sum', y_h.sum()
                #y_t = um.y_target_x(W, b, x, o, y_selected, f)
                #print 'feedback', f
                #print 'y_target', y_t
                l = um.get_loss(W, b, x, o, y_selected, f)
                print 'loss', l, ptype, fr, en_selected, fb
                dW, db = um.get_grad(W, b, x, o, y_selected, f)
                #print 'dW', dW, dW.shape
                #print 'db', db, db.shape 
                W -= 0.5 * dW
                b -= 0.5 * db
            pdb.set_trace()
