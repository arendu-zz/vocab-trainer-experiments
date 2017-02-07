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
    opt.add_argument('-r', action='store', dest='regularization', default=0.1, type=float)
    opt.add_argument('-u', action='store', dest='grad_update', default="sgd", required=True)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ = read_data('./data/data_splits/dev.data', dh)
    theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    _decay = 0.001
    _learning_rate = 0.3 #only used for sgd
    sll = SimpleLoglinear(dh, regularization = options.regularization / 100.0)
    _params = sll.get_params()
    for epoch_idx in xrange(75):
        lr = _learning_rate * (1.0  / (1.0 + _decay * epoch_idx))
        shuffle_ids = np.random.choice(xrange(len(TRAINING_SEQ)), len(TRAINING_SEQ), False)
        sys.stderr.write('-')
        for r_idx in shuffle_ids:
            sys.stderr.write('.')
            _X, _Y, _YT, _O, _S = TRAINING_SEQ[r_idx]
            if options.grad_update == "rms":
                seq_losses, seq_thetas, seq_y_hats = sll.do_rms_update(_X, _Y, _O, floatX(_S), theta_0, 0.1)
            elif options.grad_update == "sgd":
                seq_losses, seq_thetas, seq_y_hats = sll.do_sgd_update(_X, _Y, _O, floatX(_S), theta_0, lr)
            else:
                raise Exception("unknown grad update:" + options.grad_update)
            _params = sll.get_params()
            _params = np.array(_params)
            if np.isnan(seq_losses):
                raise Exception("loss is nan")
            if np.isnan(seq_y_hats).any():
                raise Exception("y_hat has nan")
            if np.isnan(_params).any():
                raise Exception("_params is nan")

        user_mean_dev_loss = []
        user_mean_p_y_u = []
        user_mean_p_y_u_c = []
        user_mean_p_y_u_ic = []
        user_traces = []
        param_str = np.array2string(_params, formatter={'float_kind':lambda p: "%.3f" % p})
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devYT, _devO, _devS = DEV_SEQ[dev_idx]
            _devS = floatX(_devS)
            mean_dev_loss = sll.get_seq_loss(_devX, _devY, _devO, _devS, theta_0)
            dev_losses = sll.get_seq_losses(_devX, _devY, _devO, _devS, theta_0)
            dev_y_hats = sll.get_seq_y_hats(_devX, _devY, _devO, _devS, theta_0)
            log_dev_y_hats = np.log(dev_y_hats)
            p_y_all = dev_y_hats[_devY == 1] #probs of all the selections
            idx_u = np.arange(_devS.shape[0])[_devS[:,(4,5,6)].any(axis=1)] #index of col when 4,5,6 is 1
            idx_u_c = np.arange(_devS.shape[0])[_devS[:,(4,7)].any(axis=1)] #index of col when 4 or 7 is 1 i.e. correct
            idx_u_ic = np.arange(_devS.shape[0])[_devS[:,(5,8)].any(axis=1)] #index of col when 5 is 1 i.e. incorrect

            p_y_u = p_y_all[idx_u] #models prob on all of users answers
            p_y_u_c = p_y_all[idx_u_c] #models pron on all of users correct answers
            p_y_u_ic = p_y_all[idx_u_ic] #models prob on all of users incorrect answers
            user_mean_p_y_u.append(p_y_u.mean())
            user_mean_p_y_u_c.append(p_y_u_c.mean())
            user_mean_p_y_u_ic.append(p_y_u_ic.mean())
            #user_plot = np.concatenate((idx_u[:,np.newaxis], p_y_u[:, np.newaxis], f_u[:,np.newaxis]), axis=1)
            #user_traces.append(user_plot)
            user_mean_dev_loss.append(mean_dev_loss)
        msg = "mean_loss:"  + "%.3f" % np.mean(user_mean_dev_loss) +\
            " model on u:" +"%.3f" % np.mean(user_mean_p_y_u) +\
            " model on c:" + "%.3f" % np.mean(user_mean_p_y_u_c) +\
            " model on ic:" + "%.3f" % np.mean(user_mean_p_y_u_ic) +\
            " params: " + param_str
        sys.stdout.write(msg +'\n')
        sys.stdout.flush()
