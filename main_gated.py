#!/usr/bin/env python
import argparse
import sys
import codecs
import numpy as np
import theano
import pdb
from code.datahelper import DataHelper
from code.gated_loglinear import GatedLogLinear
from scipy.stats.mstats import rankdata as get_ranks
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='feature', default='p.w.c')
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float)
    opt.add_argument('-t', action='store', dest='regularization_type', default='l2')
    opt.add_argument('-l', action='store', dest='learning_rate', default=1.8, type=float)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    #data_lines = codecs.open('./data/data_splits/train.data', 'r', 'utf8').readlines()
    data_lines = []
    data_lines += codecs.open('./data/data_splits/train.data', 'r', 'utf8').readlines()
    reg = options.regularization / float(len(data_lines))
    #um = UserModel(events_file, feats_file, actions_file, reg, options.regularization_type)
    dh = DataHelper(events_file, feats_file, actions_file)
    user2ave_prob = {}
    ave_prob = []
    user2ave_rank = {}
    prev_user = None
    W = None
    X, Y, O, F = [],[],[],[]
    #b = None
    SEQ = []
    for line in data_lines:
        user, uts, ptype, tstep, a_idx, fr, en_options, en_selected, fb  = [i.strip() for i in line.split('\t')]
        if user != prev_user:
            SEQ.append((np.array(X, dtype=np.int32),np.array(Y, dtype=np.int32),np.array(O, dtype=np.int32),np.array(F, dtype=np.int32)))
            X = [] #clear out seq
            Y = []
            O = []
            F = []
        if en_selected != "NO_ANSWER_MADE":
            x = dh.f2id[fr] #x
            e_id = dh.e2id[en_selected] #y index
            y_selected = np.zeros((dh.E_SIZE,)) #y
            y_selected[e_id] = 1.0 #set what the user selected to 1, and the rest to zero
            if en_options == "ALL":
                o = np.ones((dh.E_SIZE,)).astype(floatX)
            else:
                o = np.zeros((dh.E_SIZE,)).astype(floatX)
                for os in en_options.split(','):
                    o_id = dh.e2id[os.strip()]
                    o[o_id] = 1.0
            f = 0 if fb == 'incorrect' else 1
            o = o.astype(intX)
            y_selected = y_selected.astype(intX)
            #f = f.astype(intX)

            X.append(x)
            Y.append(y_selected)
            O.append(o)
            F.append(f)
        prev_user = user
    gll = GatedLogLinear(dh)
    theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    SEQ.pop(0)
    for _ in xrange(500):
        r_idx = np.random.choice(xrange(len(SEQ)), 1)[0]
        _X, _Y, _O, _F = SEQ[r_idx]
        seq_loss, seq_thetas = gll.do_update(_X, _Y, _O, _F, theta_0, 0.6)
        print seq_loss, np.min(seq_thetas), np.max(seq_thetas)
