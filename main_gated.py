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

def pad_start(_s):
    z = np.zeros((1, _s.shape[1])).astype(np.int32)
    sm1 = np.concatenate((z, _s), axis=0)
    sm1 = sm1[:_s.shape[0], :]
    sm1 = np.int32(sm1)
    return sm1


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='feature', default='p.w.pre.suf.c')
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float, required=True)
    opt.add_argument('-u', action='store', dest='grad_update', default="sgd", required=True)
    opt.add_argument('-m', action='store', dest='model', default="diag", required=True)
    opt.add_argument('--fl', action='store_true', dest='filter_loss', default=False)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ= read_data('./data/data_splits/dev.data', dh)
    assert len(TRAINING_SEQ) == 100
    assert len(DEV_SEQ) == 20
    if options.model == "diag":
        gll = GatedLogLinear(dh, regularization = options.regularization / 100.0, diag = True, filter_loss = options.filter_loss)
    elif options.model == "lowrank":
        gll = GatedLogLinear(dh, regularization = options.regularization / 100.0, diag = False, filter_loss = options.filter_loss)
    else:
        raise Exception("unknown model choice:" + options.model)
    theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    _decay = 0.001
    _learning_rate = 0.6 #only used for sgd
    for epoch_idx in xrange(75):
        lr = _learning_rate * (1.0  / (1.0 + _decay * epoch_idx))
        shuffle_ids = np.random.choice(xrange(len(TRAINING_SEQ)), len(TRAINING_SEQ), False)
        sys.stderr.write('-')
        for r_idx in shuffle_ids:
            sys.stderr.write('.')
            _X, _Y, _O, _F, _S = TRAINING_SEQ[r_idx]
            _SM1 = pad_start(_S)
            if options.grad_update == "rms":
                seq_losses, seq_thetas, seq_y_hats = gll.do_rms_update(_X, _Y, _O, _F, floatX(_S), floatX(_SM1), theta_0, 0.1)
            elif options.grad_update == "sgd":
                seq_losses, seq_thetas, seq_y_hats = gll.do_sgd_update(_X, _Y, _O, _F, floatX(_S), floatX(_SM1), theta_0, lr)
            else:
                raise Exception("unknown grad update:" + options.grad_update)
            if np.isnan(seq_losses):
                raise Exception("loss is nan")
            if np.isnan(seq_y_hats).any():
                raise Exception("y_hat has nan")
            #print r_idx, seq_losses
        user_mean_dev_loss = []
        user_mean_p_y_u = []
        user_mean_p_y_c = []
        user_mean_p_y_ic = []
        user_traces = []
        for dev_idx in xrange(len(DEV_SEQ)):
            _devX, _devY, _devO, _devF, _devS = DEV_SEQ[dev_idx]
            _devSM1 = pad_start(_devS)
            mean_dev_loss = gll.get_seq_loss(_devX, _devY, _devO, _devF, floatX(_devS), floatX(_devSM1), theta_0)
            dev_y_hats = gll.get_seq_y_hats(_devX, _devY, _devO, _devF, floatX(_devS), floatX(_devSM1), theta_0)
            p_y_all = dev_y_hats[_devY == 1]
            idx_u = np.arange(_devS.shape[0])[_devS[:,(4,5)].any(axis=1)] #index of row when 4 or 5 is 1
            idx_u_c = np.arange(_devS.shape[0])[_devS[:,(4,)].any(axis=1)] #index of row when 4 is 1 i.e. correct
            idx_u_ic = np.arange(_devS.shape[0])[_devS[:,(5,)].any(axis=1)] #index of row when 5 is 1 i.e. incorrect
            p_y_u = p_y_all[_devS[:,(4,5)].any(axis=1)] #select prob is either col 4 or 5 is 1 in T
            p_y_u_c = p_y_all[idx_u_c]
            p_y_u_ic = p_y_all[idx_u_ic]
            f_u = _devS[:,4][_devS[:,(4,5)].any(axis=1)] #select the 4th col if either col 4 or 5 is 1
            user_mean_p_y_c.append(p_y_u_c.mean())
            user_mean_p_y_ic.append(p_y_u_ic.mean())
            user_mean_p_y_u.append(p_y_u.mean())
            user_plot = np.concatenate((idx_u[:,np.newaxis], p_y_u[:, np.newaxis], f_u[:,np.newaxis]), axis=1)
            user_traces.append(user_plot)

            user_mean_dev_loss.append(mean_dev_loss)
        msg = "mean_loss:"  + "%.3f" % np.mean(user_mean_dev_loss) + " mean_p_u:" +"%.3f" % np.mean(user_mean_p_y_u) + " mean_p_c:" + "%.3f" % np.mean(user_mean_p_y_c) + " mean_p_ic:" + "%.3f" % np.mean(user_mean_p_y_ic) 
        sys.stdout.write(msg +'\n')
        sys.stdout.flush()
