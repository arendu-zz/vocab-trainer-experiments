#!/usr/bin/env python
import argparse
import sys
import codecs
import numpy as np
import theano
from code.data_reader import read_data
from code.datahelper import DataHelper
from code.recurrent_loglinear import RecurrentLoglinear
from code.eval_tools import disp_eval
#import pdb

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64


def load_json_model(file_path):
    saved_model_attrs = file_path.split('/')[-1].split('.')
    grad_update = saved_model_attrs[saved_model_attrs.index('u') + 1]
    regularization = float(saved_model_attrs[saved_model_attrs.index('r') + 1])
    grad_transform = saved_model_attrs[saved_model_attrs.index('gt') + 1]
    grad_model = saved_model_attrs[saved_model_attrs.index('gm') + 1]
    model = saved_model_attrs[saved_model_attrs.index('m') + 1]
    temp = 't0' #saved_model_attrs[saved_model_attrs.index('m') + 1]
    bl = 0.0
    do_clip = 'free'
    top_k = 'top_all'
    recurrent_ll = RecurrentLoglinear(dh, 
                    u = grad_update,
                    reg = (regularization / len(TRAINING_SEQ)), 
                    grad_transform = grad_transform,
                    learning_model = model,
                    grad_model = grad_model,
                    clip = do_clip,
                    temp_model = temp,
                    grad_top_k = top_k,
                    interpolate_bin_loss = bl,
                    saved_weights = options.saved_model)
    loaded_msg_d, loaded_dl, loaded_dpu, loaded_dacc = disp_eval(DEV_SEQ, recurrent_ll, dh, None, None)
    print 'loaded_dev:', loaded_msg_d
    return recurrent_ll

def get_observation(action_idx, theta_tm1_j):
    print dh.actions[action_idx]
    a_type, _x_t, _yt_t, _o_t = dh.action_vectors[action_idx]
    assert type(a_type) == np.ndarray
    assert a_type.shape == (10,)
    print a_type
    if a_type[[0,3]].any():
        print 'example correct'
        y_selected_j = _yt_t
        assert a_type[3] == a_type[0] == 1 # must not be a example or a revealed
    else:
        y_hat_j = rll_j.get_step_y_hat(_x_t, _o_t, theta_tm1_j)
        y_hat_j = np.reshape(y_hat_j, (dh.E_SIZE,))
        y_selected_j = np.random.choice(y_hat_j.shape[0], 1, p=y_hat_j)[0]
        print 'y_hat', y_hat_j, 'y_true', _yt_t, 'y_selected', y_selected_j
        a_type[4] = 1 if y_selected_j == _yt_t else 0
        a_type[5] = 0 if y_selected_j == _yt_t else 1
        print 'selection', 'correct' if a_type[4] == 1 else 'incorrect'
    context_t = a_type
    return y_selected_j, context_t

if __name__ == '__main__':
    np.random.seed(124)
    sys.setrecursionlimit(50000)
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='feature', default='p.w.pre.suf.c')
    opt.add_argument('--simulated_K_file', action='store', dest='K_saved_models', required=True)
    opt.add_argument('--simulated_J_file', action='store', dest='J_saved_models', required=True)
    opt.add_argument('--train', action='store', dest='training_data', default='./data/data_splits/train.data', required=False)
    opt.add_argument('--dev', action='store', dest='dev_data', default='./data/data_splits/dev.data', required=False)
    opt.add_argument('--test', action='store', dest='test_data', default='./data/data_splits/test.data', required=False)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data(options.training_data, dh)
    DEV_SEQ = read_data(options.dev_data, dh)
    T_SEQ = read_data(options.test_data, dh)
    rll_Ks = [load_json_model(l) for l in open(options.simulated_K_files, 'r', 'utf8').readlines() if l.strip() != '']
    theta_tm1_Ks = [np.zeros((dh.FEAT_SIZE,)) for _ in rll_Ks]
    b_K = (1.0 / len(rll_Ks))  * np.ones(len(rll_Ks))
    simulated_J_models = [load_json_model(l) for l in open(options.simulated_J_files, 'r', 'utf8').readlines() if l.strip() != '']
    rll_j = np.random.choice(simulated_J_models, 1)[0]
    theta_tm1_j = np.zeros(dh.FEAT_SIZE,).astype(floatX)
    context_tm1 = np.zeros(10,).astype(floatX)
    for _ in range(4): #until convergence
        action_idx = int(raw_input('pick action 0-' + str(len(dh.actions) - 1))) #15 #np.random.choice(len(dh.action_vectors), 1)[0]
        a_type, _x_t, _yt_t, _o_t = dh.action_vectors[action_idx]
        y_selected_j, context_t = get_observation(action_idx, theta_tm1_j)
        _y_t_j = np.zeros(dh.E_SIZE,).astype(np.float32)
        _y_t_j[y_selected_j] = 1.0
        for k_idx, (rll_k, theta_tm1_k) in enumerate(zip(rll_Ks, theta_tm1_Ks)):
            _theta_t_k, _g_r_k, _g_z_k = rll_k.get_step_transition(_x_t, _y_t_j.astype(floatX), _o_t.astype(floatX), context_t.astype(floatX), context_tm1.astype(floatX), theta_tm1_k.astype(floatX))
            theta_tm1_Ks[k_idx] = _theta_t_k
        theta_t_j = rll_j.get_step_transition(_x_t, _y_t_j.astype(floatX), _o_t.astype(floatX), context_t.astype(floatX), context_tm1.astype(floatX), theta_tm1_j.astype(floatX))
        #TODO: all the q-learning stuff comes here!
        context_tm1 = context_t
