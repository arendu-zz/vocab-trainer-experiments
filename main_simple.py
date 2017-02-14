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
from code.eval_tools import disp_eval, pad_start

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
    opt.add_argument('-r', action='store', dest='reg', default=0.01, type=float, required=True)
    opt.add_argument('--ur', action='store', dest='learner_reg', default=0.1, type=float, required=True)
    opt.add_argument('--bl', action='store', dest='interpolate_bin_loss', default=0.5, type=float, required=True)
    opt.add_argument('--sl', action='store', dest='use_sum_loss', default=1, type=int, required=True)
    opt.add_argument('-u', action='store', dest='grad_update', default="sgd", required=False)
    opt.add_argument('-m', action='store', dest='model', default="m0", required=True)
    opt.add_argument('-c', action='store', dest='clip', default="free", required=False)
    opt.add_argument('--st', action='store', dest='save_trace', default=None)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ = read_data('./data/data_splits/dev.data', dh)
    _theta_0 = np.zeros((dh.FEAT_SIZE,)).astype(floatX)
    _decay = 0.001
    _learning_rate = 0.5 #only used for sgd
    _clip = options.clip == "clip"
    sll = SimpleLoglinear(dh, 
                        reg = (options.reg / len(TRAINING_SEQ)), 
                        learner_reg = options.learner_reg,
                        learning_model = options.model,
                        clip = _clip,
                        use_sum_loss = options.use_sum_loss,
                        interpolate_bin_loss = options.interpolate_bin_loss)
    for epoch_idx in xrange(10 if options.model == "m0" else 100):
        lr = _learning_rate * (1.0  / (1.0 + _decay * epoch_idx))
        shuffle_ids = np.random.choice(xrange(len(TRAINING_SEQ)), len(TRAINING_SEQ), False)
        sys.stderr.write('-')
        for r_idx in shuffle_ids[:]:
            sys.stderr.write('.')
            _X, _Y, _YT, _O, _S = TRAINING_SEQ[r_idx]
            _SM1 = pad_start(_S)
            if options.grad_update == "rms":
                seq_losses, seq_thetas, seq_y_hats = sll.do_rms_update(_X, _Y, _YT, _O, _S, _SM1, _theta_0, 1.0)
            elif options.grad_update == "sgd":
                seq_losses, seq_thetas, seq_y_hats = sll.do_sgd_update(_X, _Y, _YT, _O, _S, _SM1, _theta_0, lr)
            else:
                raise Exception("unknown grad update:" + options.grad_update)
            _params = sll.get_params()
            _max_p = []
            if np.isnan(seq_losses):
                raise Exception("loss is nan")
            if np.isnan(seq_y_hats).any():
                raise Exception("y_hat has nan")
            for _p in _params:
                if np.isnan(_p).any():
                    raise Exception("_params is nan")
                _max_p.append(np.max(_p))
        print 'dev'
        disp_eval(DEV_SEQ, sll, dh, options.save_trace, epoch_idx) 
        print 'train'
        disp_eval(TRAINING_SEQ[:2], sll, dh, None, None)
