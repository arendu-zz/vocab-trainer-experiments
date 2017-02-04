#!/usr/bin/env python
import argparse
import sys
import codecs
import numpy as np
import theano
from code.data_reader import read_data
from code.datahelper import DataHelper
from code.gated_loglinear import GatedLogLinear
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
    opt.add_argument('-f', action='store', dest='feature', default='p.w.l.pre.suf.c')
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ= read_data('./data/data_splits/dev.data', dh)
    gll = GatedLogLinear(dh, 0.001)
    theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    for _ in xrange(25):
        shuffle_ids = np.random.choice(xrange(len(TRAINING_SEQ)), len(TRAINING_SEQ), False)
        for r_idx in shuffle_ids:
            _X, _Y, _O, _F = TRAINING_SEQ[r_idx]
            seq_loss, seq_thetas, seq_y_hats = gll.do_sgd_update(_X, _Y, _O, _F, theta_0, 0.1)
            if np.isnan(seq_loss):
                raise Exception("loss is nan")
            if np.isnan(seq_y_hats).any():
                raise Exception("y_hat has nan")
            print r_idx, seq_loss
        sum_dev_losses = 0.0
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devO, _devF = DEV_SEQ[r_idx]
            dev_loss = gll.get_seq_loss(_devX, _devY, _devO, _devF, theta_0)
            sum_dev_losses += dev_loss
        print 'dev:',sum_dev_losses
        #TODO:save params
