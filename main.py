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
    f = np.array([1,1,0,0]) #2 questions correct 2 questions wrong...
    print um.dh.E_SIZE
    print um.dh.F_SIZE
    print um.dh.FEAT_SIZE
    for line in codecs.open('./data/data_splits/group0.data', 'r', 'utf8').readlines():
        user, uts, ptype, tstep, a_idx, fr, en_options, en_selected, fb  = [i.strip() for i in line.split('\t')]
        if en_selected != "NO_ANSWER_MADE":
            x = np.array([um.dh.f2id[fr]]) #x
            e_id = um.dh.e2id[en_selected.lower()] #y
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
            y_d = um.y_dot_x(x.astype(np.int64)) 
            print y_d
            y_dm = um.y_dot_masked_x(x.astype(np.int64), o.astype(floatX)) 
            print y_dm
            y_t = um.y_target_x(x.astype(np.int64), o.astype(floatX), y_selected.astype(floatX), f.astype(np.int64))
            print 'feedback', f
            print y_t
