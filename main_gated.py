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
    opt.add_argument('-f', action='store', dest='feature', default='p.w.pre.suf.c')
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float, required=True)
    opt.add_argument('-u', action='store', dest='grad_update', default="sgd", required=True)
    opt.add_argument('-m', action='store', dest='model', default="diag", required=True)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ= read_data('./data/data_splits/dev.data', dh)
    if options.model == "diag":
        gll = GatedLogLinear(dh, regularization = options.regularization / 100.0, diag = True)
    elif options.model == "lowrank":
        gll = GatedLogLinear(dh, regularization = options.regularization / 100.0, diag = False)
    else:
        raise Exception("unknown model choice:" + options.model)
    theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    _decay = 0.001
    _learning_rate = 0.5 #only used for sgd
    for epoch_idx in xrange(50):
        lr = _learning_rate * (1.0  / (1.0 + _decay * epoch_idx))
        shuffle_ids = np.random.choice(xrange(len(TRAINING_SEQ)), len(TRAINING_SEQ), False)
        sys.stderr.write('-')
        for r_idx in shuffle_ids:
            sys.stderr.write('.')
            _X, _Y, _O, _F = TRAINING_SEQ[r_idx]
            if options.grad_update == "rms":
                seq_losses, seq_thetas, seq_y_hats = gll.do_rms_update(_X, _Y, _O, _F, theta_0, 0.1)
            elif options.grad_update == "sgd":
                seq_losses, seq_thetas, seq_y_hats = gll.do_sgd_update(_X, _Y, _O, _F, theta_0, lr)
            else:
                raise Exception("unknown grad update:" + options.grad_update)
            if np.isnan(seq_losses):
                raise Exception("loss is nan")
            if np.isnan(seq_y_hats).any():
                raise Exception("y_hat has nan")
            #print r_idx, seq_losses
        sum_dev_losses = 0.0
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devO, _devF = DEV_SEQ[dev_idx]
            dev_loss = gll.get_seq_loss(_devX, _devY, _devO, _devF, theta_0)
            sum_dev_losses += dev_loss
        msg = str(epoch_idx) + " dev:" + str(sum_dev_losses)
        sys.stdout.write(msg +'\n')
        sys.stdout.flush()
        #TODO:save params
