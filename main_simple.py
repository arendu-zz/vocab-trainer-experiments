#!/usr/bin/env python
import argparse
import sys
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
    assert len(TRAINING_SEQ) == 100
    DEV_SEQ = read_data('./data/data_splits/dev.data', dh)
    assert len(DEV_SEQ) == 20
    lrs = np.arange(16.0) * 0.1 if options.update_method == "simple" else [0]
    for _lr in lrs: 
        _lr = floatX(_lr)
        if options.update_method in ["simple", "adagrad", "adadelta", "rmsprop"]:
            sll = SimpleLoglinear(dh, _lr, adapt = options.update_method)
        else:
            raise Exception("unknown update_method:" + options.update_method)
        theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
        E_g_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
        E_dx_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
        sum_dev_losses = 0.0
        ave_p_y = 0.0
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devO, _devF, _devT = DEV_SEQ[dev_idx]
            dev_loss = sll.get_seq_loss(_devX, _devY, _devO, _devF, theta_0, E_g_0, E_dx_0)
            dev_y_hats = sll.get_seq_y_hats(_devX, _devY, _devO, _devF, theta_0, E_g_0, E_dx_0)
            p_y = dev_y_hats[_devY == 1]
            p_y = p_y[_devT[:,(0,3)].sum(axis=1) == 0]
            ave_p_y += np.mean(p_y)
            sum_dev_losses += dev_loss
        msg = "learning rate:" + str(sll.lr) + " ave_dev_loss per seq:" + str(sum_dev_losses / len(DEV_SEQ)) + "sum_ave_p_y:" + str(ave_p_y)
        sys.stdout.write(msg +'\n')
        sys.stdout.flush()
