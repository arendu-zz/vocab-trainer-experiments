#!/usr/bin/env python
import argparse
import pdb
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
        user_mean_dev_loss = []
        user_mean_p_y_u = []
        user_mean_p_y_c = []
        user_mean_p_y_ic = []
        user_traces = []
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devO, _devF, _devT = DEV_SEQ[dev_idx]
            mean_dev_loss = sll.get_seq_loss(_devX, _devY, _devO, _devF, theta_0, E_g_0, E_dx_0)
            dev_y_hats = sll.get_seq_y_hats(_devX, _devY, _devO, _devF, theta_0, E_g_0, E_dx_0)
            p_y_all = dev_y_hats[_devY == 1]
            idx_u = np.arange(_devT.shape[0])[_devT[:,(4,5)].any(axis=1)] #index of row when 4 or 5 is 1
            idx_u_c = np.arange(_devT.shape[0])[_devT[:,(4,)].any(axis=1)] #index of row when 4 is 1 i.e. correct
            idx_u_ic = np.arange(_devT.shape[0])[_devT[:,(5,)].any(axis=1)] #index of row when 5 is 1 i.e. incorrect
            p_y_u = p_y_all[_devT[:,(4,5)].any(axis=1)] #select prob is either col 4 or 5 is 1 in T
            p_y_u_c = p_y_all[idx_u_c]
            p_y_u_ic = p_y_all[idx_u_ic]
            f_u = _devT[:,4][_devT[:,(4,5)].any(axis=1)] #select the 4th col if either col 4 or 5 is 1
            user_mean_p_y_c.append(p_y_u_c.mean())
            user_mean_p_y_ic.append(p_y_u_ic.mean())
            user_mean_p_y_u.append(p_y_u.mean())
            user_plot = np.concatenate((idx_u[:,np.newaxis], p_y_u[:, np.newaxis], f_u[:,np.newaxis]), axis=1)
            user_traces.append(user_plot)

            user_mean_dev_loss.append(mean_dev_loss)
        msg = "learning rate:" + "%.3f" % sll.lr + " mean_loss:"  + "%.3f" % np.mean(user_mean_dev_loss) + " mean_p_u:" +"%.3f" % np.mean(user_mean_p_y_u) + " mean_p_c:" + "%.3f" % np.mean(user_mean_p_y_c) + " mean_p_ic:" + "%.3f" % np.mean(user_mean_p_y_ic) 
        sys.stdout.write(msg +'\n')
        sys.stdout.flush()
