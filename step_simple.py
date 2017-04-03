#dd_!/usr/bin/env python
import argparse
import sys
import codecs
import numpy as np
import theano
from code.data_reader import read_data
from code.datahelper import DataHelper
from code.recurrent_loglinear import RecurrentLoglinear
from code.rewards import Reward
from code.dqn import DQNTheano as DQN
#from code.eval_tools import disp_eval
import pdb

__author__ = 'arenduchintala'

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64

global term_t_exp, B_t_exp, A_t_exp, R_t_exp, B_tm1_exp
term_t_exp = None
B_t_exp = None
A_t_exp = None
R_t_exp = None
B_tm1_exp = None

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

def merge(belief_weights, states):
    b = np.zeros_like(states[0])
    for k_idx, (bw, s) in enumerate(zip(belief_weights, states)):
        b += bw * s
    return b

def add_to_experince(term_t, B_t, a_t, r_t, B_tm1):
    global term_t_exp, B_t_exp, B_tm1_exp, A_t_exp, R_t_exp
    if term_t_exp is None:
        term_t_exp = np.asarray([term_t])
    else:
        term_t_exp = np.hstack((term_t_exp, term_t))

    if B_t_exp is None:
        B_t_exp = B_t
        B_t_exp = np.reshape(B_t_exp, (1, B_t.shape[0]))
    else:
        B_t_exp = np.vstack((B_t_exp, B_t))

    if A_t_exp is None:
        A_t_exp = np.asarray([a_t])
    else:
        A_t_exp = np.hstack((A_t_exp, a_t))

    if R_t_exp is None:
        R_t_exp = np.asarray([r_t])
    else:
        R_t_exp = np.hstack((R_t_exp, r_t))

    if B_tm1_exp is None:
        B_tm1_exp = B_tm1
        B_tm1_exp = np.reshape(B_tm1_exp, (1, B_tm1.shape[0]))
    else:
        B_tm1_exp = np.vstack((B_tm1_exp, B_tm1))
    assert term_t_exp.shape[0] == B_t_exp.shape[0] == A_t_exp.shape[0] == R_t_exp.shape[0]
    return True


def get_batch_experience(bs = 100):
    global term_t_exp, B_t_exp, B_tm1_exp, A_t_exp, R_t_exp
    bs = bs if B_t_exp.shape[0] > bs else B_t_exp.shape[0]
    exp_ids = np.random.choice(B_t_exp.shape[0], bs, False)
    B_t_sampled = B_t_exp[exp_ids, :]
    B_t_sampled = B_t_sampled.astype(floatX)
    B_tm1_sampled = B_tm1_exp[exp_ids, :]
    B_tm1_sampled = B_tm1_sampled.astype(floatX)
    term_t_sampled = term_t_exp[exp_ids]
    term_t_sampled = term_t_sampled.astype(np.int64)
    R_t_sampled =R_t_exp[exp_ids]
    R_t_sampled = R_t_sampled.astype(floatX)
    A_t_sampled = A_t_exp[exp_ids]
    A_t_sampled = A_t_sampled.astype(np.int64)
    return term_t_sampled, B_t_sampled, A_t_sampled, R_t_sampled, B_tm1_sampled

def remove_from_experince(limit = 10000):
    global term_t_exp, B_t_exp, B_tm1_exp, A_t_exp, R_t_exp
    if B_t_exp is not None and B_t_exp.shape[0] >= limit:
        remove_ids = np.random.choice(B_t_exp.shape[0], B_t_exp.shape[0] - limit, False)
        B_t_exp = np.delete(B_t_exp, remove_ids, axis = 0)
        B_tm1_exp = np.delete(B_tm1_exp, remove_ids, axis = 0)
        term_t_exp = np.delete(term_t_exp, remove_ids, axis = 0)
        A_t_exp = np.delete(A_t_exp, remove_ids, axis = 0)
        R_t_exp = np.delete(R_t_exp, remove_ids, axis = 0)
    else:
        pass
    return True


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
    opt.add_argument('--epochs', action='store', dest='EPOCHS', default=100, type=int)
    opt.add_argument('--improvement', action='store_true', dest='IMPROVEMEMT', default=False)
    options = opt.parse_args()

    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    quiz_actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions.test'

    dh = DataHelper(events_file, feats_file, actions_file, quiz_actions_file)
    reward = Reward(dh, reward_type='match')
    dqn = DQN([dh.FEAT_SIZE, np.int64(0.33 * dh.FEAT_SIZE), np.int64(0.1 * dh.FEAT_SIZE), len(dh.actions)], gamma = 0.8)
    TRAINING_SEQ = read_data(options.training_data, dh)
    DEV_SEQ = read_data(options.dev_data, dh)
    T_SEQ = read_data(options.test_data, dh)
    K_files = codecs.open(options.K_saved_models, 'r', 'utf8').readlines()
    J_files = codecs.open(options.J_saved_models, 'r', 'utf8').readlines()
    print 'loading Ks:'
    rll_Ks = [load_json_model(l.strip()) for l in K_files[:] if l.strip() != '']
    print 'loading Js'
    rll_Js = [load_json_model(l.strip()) for l in J_files[:] if l.strip() != '']
     
    eps = 0.9

    rpe_list = np.zeros(options.EPOCHS)
    qs_list = np.zeros(options.EPOCHS)
    lpe_list = np.zeros(options.EPOCHS)
    for epoch_idx in xrange(100):
        rll_j = np.random.choice(rll_Js, 1)[0]
        print 'sampled:', rll_j.saved_weights, 'jth model'
        theta_tm1_j = np.zeros(dh.FEAT_SIZE,).astype(floatX)
        context_tm1 = np.zeros(10,).astype(floatX)
        r_tm1 = reward.get_reward(rll_j, theta_tm1_j)
        rpe = 0.0
        lpe = 0.0 #loss per episode (accumilated over the episode)
        quiz_score = 0.0
        eps = 0.3 if eps < 0.3 else np.power(eps, epoch_idx)
        remove_from_experince()
        theta_tm1_Ks = [np.zeros((dh.FEAT_SIZE,)).astype(floatX) for _ in rll_Ks]
        theta_t_Ks = [None] * len(theta_tm1_Ks)
        belief_weight_Ks = (1.0 / len(rll_Ks))  * np.ones(len(rll_Ks))
        B_tm1 = merge(belief_weight_Ks, theta_tm1_Ks)
        for t in range(35): # each training step t = [1:T]
            if np.random.rand() < eps:
                sys.stderr.write('.')
                action_idx_t = np.random.choice(len(dh.actions), 1)[0] #int(raw_input('pick action 0-' + str(len(dh.actions) - 1))) #15 #np.random.choice(len(dh.action_vectors), 1)[0]
            else:
                sys.stderr.write('*')
                q_hat = dqn.get_Q_hat(B_tm1)
                action_idx_t = np.argmax(q_hat)

            a_type, _x_t, _yt_t, _o_t = dh.action_vectors[action_idx_t]

            y_sel_j, y_sel_j_vec, y_hat_j, context_t = get_observation(action_idx_t, rll_j, theta_tm1_j, disp=True)
            theta_t_j, g_r_j, g_z_j = rll_j.get_step_transition(_x_t, y_sel_j_vec.astype(floatX), _o_t.astype(floatX), context_t.astype(floatX), context_tm1.astype(floatX), theta_tm1_j.astype(floatX))
            term_t = None
            if t < 34: 
                # non-terminal state
                term_t = 0
                reward_Ks = np.zeros(len(rll_Ks),)
                for k_idx, (rll_k, theta_tm1_k) in enumerate(zip(rll_Ks, theta_tm1_Ks)):
                    _, _, y_hat_k, _ = get_observation(action_idx_t, rll_k, theta_tm1_k)
                    belief_weight_Ks[k_idx] = (y_hat_k[y_sel_j] * belief_weight_Ks[k_idx]) + 1e-8
                    _theta_t_k, _g_r_k, _g_z_k = rll_k.get_step_transition(_x_t, y_sel_j_vec.astype(floatX), _o_t.astype(floatX), context_t.astype(floatX), context_tm1.astype(floatX), theta_tm1_k.astype(floatX))
                    theta_t_Ks[k_idx] = _theta_t_k
                    _reward_t_k = reward.get_reward(rll_k, _theta_t_k)
                    reward_Ks[k_idx] = _reward_t_k
                    print '\t', '.'.join(rll_k.saved_weights.split('/')[-1].split('.')[-4:]), k_idx, ' prob on obs=', y_hat_k[y_sel_j], 'reward=', _reward_t_k
                #end for k = 0 to K
                belief_weight_Ks = belief_weight_Ks * (1.0 / np.sum(belief_weight_Ks))
                assert np.abs(np.sum(belief_weight_Ks) - 1.0) < 1e-10
                B_t = merge(belief_weight_Ks, theta_t_Ks) 
                r_t = np.sum([b * r for b,r in zip(belief_weight_Ks, reward_Ks)])
            else:
                #terminal state
                term_t = 1
                belief_weight_Ks = belief_weight_Ks * (1.0 / np.sum(belief_weight_Ks))
                assert np.abs(np.sum(belief_weight_Ks) - 1.0) < 1e-10
                B_t = merge(belief_weight_Ks, theta_t_Ks) 
                _reward_t_j = reward.get_reward(rll_j, theta_t_j)
                r_t = _reward_t_j # at the end we can see the true score of the 'real' simulated user
                qs = reward.get_match_reward(rll_j, theta_t_j)
                pass
            context_tm1 = context_t
            theta_tm1_j = theta_t_j
            for k_idx, _theta_t_k in enumerate(theta_t_Ks):
                theta_tm1_Ks[k_idx] = _theta_t_k

            if options.IMPROVEMEMT:
                add_to_experince(term_t, B_t, action_idx_t, r_t - r_tm1, B_tm1)
                rpe += (r_t - r_tm1)
                r_tm1 = r_t
            else:
                add_to_experince(term_t, B_t, action_idx_t, r_t, B_tm1)
                rpe += r_t
            term_t_minibatch, B_t_minibatch, A_t_minibatch, R_t_minibatch, B_tm1_minibatch = get_batch_experience()
            total_loss, batch_loss, loss_vec = dqn.do_update(term_t_minibatch,
                                                B_t_minibatch, 
                                                A_t_minibatch, 
                                                R_t_minibatch, 
                                                B_tm1_minibatch,
                                            0.001)
            lpe += np.mean(loss_vec) 
            #if np.isnan(total_loss):
            #    raise Exception("total_loss is Nan")
            #else:
            #    pass
        print epoch_idx, rpe, lpe, qs
        rpe_list[epoch_idx] = rpe
        qs_list[epoch_idx] = qs
        lpe_list[epoch_idx] = lpe
