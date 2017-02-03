#!/usr/bin/env python
import random
import argparse
from code.utils import *
from code.usermodel_nn import UserModel 
import pdb
import numpy as np
import theano
import theano.tensor as T
from code.rewards import Reward
from code.datahelper import DataHelper
__author__ = 'arenduchintala'
if theano.config.floatX == 'float32':
    floatX = np.float32
else:
    floatX = np.float64

np.set_printoptions(precision=4, suppress = True)
np.seterr(all = 'raise')

def model_pred_dist(u, dh,  f_id, e_id, c_ids):
    option_ids = c_ids + [e_id] 
    y_dot = u.get_y_dot(np.array([f_id]))
    y_score_x = u.get_y_score_x(f_id)
    print y_score_x , 'is here!', y_dot, 'is y_dot'
    non_option_ids = np.delete(dh.all_e_ids, option_ids)
    y_score_x[non_option_ids] = 0.0
    mc_prob_y_given_x = y_score_x / np.sum(y_score_x)
    for _idx, e_prime_idx in enumerate(option_ids):
        print _idx, dh.id2e[e_prime_idx],'\t\t', mc_prob_y_given_x[e_prime_idx]
    return True

if __name__ == '__main__':
    random.seed(1234)
    np.random.seed(1234)
    opt= argparse.ArgumentParser(description="accepts a csv file and generates log linear features")
    #insert options here
    opt.add_argument('-e', action='store', dest='event2feats_file', default='./data/es-en-tiny.p.w.pre.suf.c.event2feats')
    opt.add_argument('-f', action='store', dest='feat2id_file', default='./data/es-en-tiny.p.w.pre.suf.c.feat2id')
    opt.add_argument('-a', action='store', dest='actions_file', default='./data/es-en-tiny.mc.actions')
    options = opt.parse_args()
    dh  = DataHelper(options.event2feats_file, options.feat2id_file, options.actions_file)
    u = UserModel(options.event2feats_file, options.feat2id_file, options.actions_file, optimizer = UserModel.RMSProp)
    #random.shuffle(dh.actions)
    actions = dh.actions[:]
    seenactions = []
    reward = Reward(dh)
    for _ in xrange(3):
        #random.shuffle(actions)
        seenactions = []
        total_llr = 0.0
        total_llb = 0.0
        total_lla = 0.0
        init_r = reward.get_accuracy_reward(u, display=True)
        for _ in xrange(len(dh.actions)):
            #print 'reward before anything displayed', init_r
            print '***************************NEW ACTION******************************'
            a_idx = int(raw_input('select action:[0-' + str(len(dh.actions) - 1) + ']'))
            print 'selected action:', dh.actions[a_idx] 
            a = dh.actions[a_idx]
            llr = reward.get_ll_reward(u)
            llb = reward.get_ll_baseline_reward()
            total_llr += llr
            #print 'REWARD at current time:', llr, llb, llm
            #print 'TOTAL REWARD:', total_llr, total_llb, total_llm
            f_id = dh.f2id[a[2]]
            e_id = dh.e2id[a[3]]
            seenactions.append(a)
            if len(a) > 4:
                c_ids = [dh.e2id[c_id] for c_id in a[4:]]
                options  = c_ids + [e_id]
                print '*********PROMPT*************'
                #random.shuffle(options)
                print f_id, dh.id2f[f_id], '\n\t', '\n\t'.join([str(idx) + ':' + dh.id2e[o_id] for idx, o_id in enumerate(options)])
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                print '****(pred) before*************'
                model_pred_dist(u, dh, f_id, e_id, c_ids)
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                choice = int(raw_input('selection?'))
                selected_str = dh.id2e[options[choice]]
                selected_id = options[choice]
                print 'selected:', selected_str
                print 'result  :', ('CORRECT!!' if selected_id == e_id else 'Wrong')
                u.update(a, observation = selected_id)
                print '********(pred) after***********'
                model_pred_dist(u, dh, f_id, e_id, c_ids)
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
            else:
                print '*********PROMPT*************'
                print dh.id2f[f_id], '::', dh.id2e[e_id]
                print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
                u.update(a)
            tmp = raw_input('next...')
            #influence_weights(u, dh)
            print '~~~~~~~~~~~~~~~~~~~~~~~~~~~END ACTION~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n'
            rr = reward.get_accuracy_reward(u, display=True)
            total_lla += rr
            print 'accuracy after this action', rr
            #print 'total reward', total_lla
            #print 'state at this point', u.get_weight_copy()
            '''
            print '******HISTORY*****'
            for idx, sa in enumerate(seenactions):
                seen_f_id = dh.f2id[sa[2]]
                seen_e_id = dh.e2id[sa[3]]
                max_id = np.argmax(u.get_y_score_x(seen_f_id))
                model_prediction = dh.id2e[max_id] 
                print idx, dh.id2f[seen_f_id], 'correct...' if max_id == seen_e_id else 'incorrect!' + dh.id2e[max_id]
                #print u.get_y_score_x(seen_f_id)[max_id] , u.get_y_score_x(seen_f_id)[seen_e_id], 'score'
            print '~~~~END HISTORY~~~~~\n'
            '''
