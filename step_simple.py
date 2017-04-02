#!/usr/bin/env python
import argparse
import sys
import codecs
import numpy as np
import theano
from code.data_reader import read_data
from code.datahelper import DataHelper
from code.recurrent_loglinear import RecurrentLoglinear
from code.rewards import Reward
#from code.eval_tools import disp_eval
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
            saved_weights = file_path)
    #loaded_msg_d, loaded_dl, loaded_dpu, loaded_dacc = disp_eval(DEV_SEQ, recurrent_ll, dh, None, None)
    #print 'loaded_dev:', loaded_msg_d
    print 'loaded', file_path
    return recurrent_ll

def get_observation(action_idx, rll, theta_tm1, disp=False):
    if disp:
        print dh.actions[action_idx]
    a_type, _x_t, _yt_t, _o_t = dh.action_vectors[action_idx]
    assert type(a_type) == np.ndarray
    assert a_type.shape == (10,)
    if disp:
        print a_type
    y_hat = rll.get_step_y_hat(_x_t, _o_t, theta_tm1)
    y_hat = np.reshape(y_hat, (dh.E_SIZE,))
    if a_type[[0,3]].any():
        if disp:
            print 'example correct'
        y_sel = _yt_t
        assert a_type[3] == a_type[0] == 1 # must not be a example or a revealed
    else:
        y_sel = np.random.choice(y_hat.shape[0], 1, p=y_hat)[0]
        a_type[4] = 1 if y_sel == _yt_t else 0
        a_type[5] = 0 if y_sel == _yt_t else 1
    if disp:
        print 'prob on true', y_hat[_yt_t], _yt_t
        print 'prob on sel', y_hat[y_sel], y_sel
        print 'selection', 'example' if a_type[3] == 1 else ('correct' if a_type[4] == 1 else 'incorrect')
    context_t = a_type
    y_sel_vec = np.zeros(dh.E_SIZE,).astype(floatX)
    y_sel_vec[y_sel] = 1.0
    return y_sel, y_sel_vec, y_hat, context_t


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
    quiz_actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions.test'
    dh = DataHelper(events_file, feats_file, actions_file, quiz_actions_file)
    reward = Reward(dh)
    TRAINING_SEQ = read_data(options.training_data, dh)
    DEV_SEQ = read_data(options.dev_data, dh)
    T_SEQ = read_data(options.test_data, dh)
    K_files = codecs.open(options.K_saved_models, 'r', 'utf8').readlines()
    J_files = codecs.open(options.J_saved_models, 'r', 'utf8').readlines()
    print 'k'
    rll_Ks = [load_json_model(l.strip()) for l in K_files[:] if l.strip() != '']
    print 'j'
    rll_Js = [load_json_model(l.strip()) for l in J_files[:] if l.strip() != '']
    theta_tm1_Ks = [np.zeros((dh.FEAT_SIZE,)).astype(floatX) for _ in rll_Ks]
    belief_weight_Ks = (1.0 / len(rll_Ks))  * np.ones(len(rll_Ks))
    #TODO: setup experience replay memory
    #while not converged
    rll_j = np.random.choice(rll_Js, 1)[0]
    print 'sampled:', rll_j.saved_weights, 'jth model'
    theta_tm1_j = np.zeros(dh.FEAT_SIZE,).astype(floatX)
    context_tm1 = np.zeros(10,).astype(floatX)
    for _ in range(10): # each training step t = [1:T]
        print 'training step', _
        action_idx = int(raw_input('pick action 0-' + str(len(dh.actions) - 1))) #15 #np.random.choice(len(dh.action_vectors), 1)[0]
        a_type, _x_t, _yt_t, _o_t = dh.action_vectors[action_idx]
        y_sel_j, y_sel_j_vec, y_hat_j, context_t = get_observation(action_idx, rll_j, theta_tm1_j, disp=True)
        theta_t_j, g_r_j, g_z_j = rll_j.get_step_transition(_x_t, y_sel_j_vec.astype(floatX), _o_t.astype(floatX), context_t.astype(floatX), context_tm1.astype(floatX), theta_tm1_j.astype(floatX))
        context_tm1 = context_t
        theta_tm1_j = theta_t_j
        for k_idx, (rll_k, theta_tm1_k) in enumerate(zip(rll_Ks, theta_tm1_Ks)):
            _, _, y_hat_k, _ = get_observation(action_idx, rll_k, theta_tm1_k)
            belief_weight_Ks[k_idx] = y_hat_k[y_sel_j] * belief_weight_Ks[k_idx]
            _theta_t_k, _g_r_k, _g_z_k = rll_k.get_step_transition(_x_t, y_sel_j_vec.astype(floatX), _o_t.astype(floatX), context_t.astype(floatX), context_tm1.astype(floatX), theta_tm1_k.astype(floatX))
            _reward_t_k = reward.get_ll_reward(rll_k, _theta_t_k)
            print '.'.join(rll_k.saved_weights.split('/')[-1].split('.')[-4:]), k_idx, ' prob on obs=', y_hat_k[y_sel_j], 'reward=', _reward_t_k
            theta_tm1_Ks[k_idx] = _theta_t_k
        belief_weight_Ks = belief_weight_Ks * (1.0 / np.sum(belief_weight_Ks))
        print np.sum(belief_weight_Ks)
        assert np.abs(np.sum(belief_weight_Ks) - 1.0) < 1e-10
