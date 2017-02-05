#!/usr/bin/env python
import argparse
import sys
import pdb
import codecs
import numpy as np
import theano
from code.data_reader import read_data
from code.datahelper import DataHelper
from code.simple_loglinear import SimpleLoglinear
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
    opt.add_argument('-f', action='store', dest='feature', default='p.w.pre.suf.c')
    opt.add_argument('-m', action='store', dest='update_method', default='simple', required = True)
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ= read_data('./data/data_splits/dev.data', dh)
    for _lr in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        if options.update_method in ["simple", "adagrad", "adadelta", "rmsprop"]:
            sll = SimpleLoglinear(dh, _lr, adapt = options.update_method)
        else:
            raise Exception("unknown update_method:" + options.update_method)
        theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
        E_g_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
        E_dx_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
        for _ in xrange(0): #nothing to learn!!
            #shuffle_ids = np.random.choice(xrange(len(TRAINING_SEQ)), len(TRAINING_SEQ), False)
            shuffle_ids = range(len(TRAINING_SEQ))
            for r_idx in shuffle_ids:
                _X, _Y, _O, _F = TRAINING_SEQ[r_idx]
                seq_losses = sll.get_seq_losses(_X, _Y, _O, _F, theta_0)
                seq_loss = sll.get_seq_loss(_X, _Y, _O, _F, theta_0)
                #print seq_losses
                print seq_loss
                seq_y_hats = sll.get_seq_y_hats(_X, _Y, _O, _F, theta_0)
                seq_thetas = sll.get_seq_thetas(_X, _Y, _O, _F, theta_0)
                if np.isnan(seq_loss):
                    raise Exception("loss is nan")
                if np.isnan(seq_y_hats).any():
                    raise Exception("y_hat has nan")
                if np.isnan(seq_thetas).any():
                    raise Exception("thetas has nan")
                print r_idx, seq_loss
        sum_dev_losses = 0.0
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devO, _devF = DEV_SEQ[dev_idx]
            dev_loss = sll.get_seq_loss(_devX, _devY, _devO, _devF, theta_0, E_g_0, E_dx_0)
            sum_dev_losses += dev_loss
        msg = "learning rate:" + str(sll.lr) + " dev:" + str(sum_dev_losses)
        sys.stdout.write(msg +'\n')
        sys.stdout.flush()
