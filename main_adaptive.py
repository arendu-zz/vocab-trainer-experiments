#!/usr/bin/env python
from code.loglinear import UserModel
import argparse
import sys
import codecs
import numpy as np
import pdb
import theano

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'
if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64

global _eps, decay, _eta
_eps = np.finfo(floatX).eps
decay = 0.9
_eta = 0.01


def rms(v):
    global _eps
    return np.sqrt(v + _eps)

def adapt_grad(adapt, g, E_g, E_dx):
    global decay, _eps, _eta
    if adapt == "adadelta":
        E_g = (decay * E_g) + ((1.0 - decay) * (g ** 2))
        E_g[np.absolute(E_g) < _eps] = 0.0
        grad = (rms(E_dx) / rms(E_g)) * g
        grad[np.absolute(grad) < _eps] = 0.0
        E_dx = (decay * E_dx) + ((1.0 - decay) * (grad ** 2))
        return grad, E_g, E_dx
    elif adapt == "adagrad":
        g[np.absolute(g) < _eps] = 0.0
        E_g += g ** 2
        E_g[np.absolute(E_g) < _eps] = 0.0
        grad = (_eta / rms(E_g)) * g
        return grad, E_g, E_dx #E_dx does nothing...
    elif adapt == "rmsprop":
        E_g = (decay * E_g) + ((1.0 - decay) * (g ** 2))
        E_g[np.absolute(E_g) < _eps] = 0.0
        grad = (_eta / rms(E_g)) * g
        return grad, E_g, E_dx
    else:
        raise Exception("unknown adapt grad")


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='feature', default='p.w.c')
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float)
    opt.add_argument('-t', action='store', dest='regularization_type', default='l2')
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    data_lines = codecs.open('./data/data_splits/train.data', 'r', 'utf8').readlines()
    data_lines += codecs.open('./data/data_splits/dev.data', 'r', 'utf8').readlines()
    reg = options.regularization / float(len(data_lines))
    decay = 0.9 #adadelta
    um = UserModel(events_file, feats_file, actions_file, reg, options.regularization_type)
    user2ave_prob = {}
    ave_prob = []
    user2ave_rank = {}
    prev_user = None
    for line in data_lines: 
        user, uts, ptype, tstep, a_idx, fr, en_options, en_selected, fb  = [i.strip() for i in line.split('\t')]
        if user != prev_user:
            sys.stderr.write('.')
            #print 'new user', user
            W = np.zeros((um.dh.FEAT_SIZE,)).astype(floatX)
            E_g = np.zeros_like(W).astype(floatX)
            E_dx = np.zeros_like(W).astype(floatX)
        else:
            pass
        if en_selected != "NO_ANSWER_MADE":
            x = np.array([um.dh.f2id[fr]]) #x
            e_id = um.dh.e2id[en_selected] #y index
            y_selected = np.zeros((1, um.dh.E_SIZE)) #y
            y_selected[0, e_id] = 1.0 #set what the user selected to 1, and the rest to zero
            if en_options == "ALL":
                o = np.ones((1, um.dh.E_SIZE)).astype(floatX)
            else:
                o = np.zeros((1, um.dh.E_SIZE)).astype(floatX)
                for os in en_options.split(','):
                    o_id = um.dh.e2id[os.strip()]
                    o[0, o_id] = 1.0
            f = np.array([0], dtype=bool) if fb == 'incorrect' else np.array([1], dtype=bool)
            x = x.astype(intX)
            o = o.astype(intX)
            y_selected = y_selected.astype(intX)
            f = f.astype(intX)

            loss  = um.get_loss(W, x, o, y_selected, f)
            if np.isnan(loss):
                raise Exception("loss is nan")
            dW = um.get_grad(W, x, o, y_selected, f)
            if np.isnan(dW).any(): 
                raise Exception("grad W has nan")
            #if np.isnan(db).any():
            #    raise Exception("grad b has nan")

            if ptype != 'EX' and fb != 'revealed':
                #print tstep, ptype, fr, en_selected, um.dh.true_f2e[fr], fb
                y_hat = um.y_given_x(W, x, o)
                if np.isnan(y_hat).any():
                    raise Exception("y_hat has nan")
                #y_hat_ranks = um.dh.E_SIZE - get_ranks(y_hat, axis=1) 
                loss  = um.get_loss(W, x, o, y_selected, f)
                p_e_selected_given_x = y_hat[0, e_id]
                #rank_e_selected_given_x = y_hat_ranks[0, e_id]
                up_list = user2ave_prob.get(user, [])
                up_list.append(p_e_selected_given_x)
                user2ave_prob[user] = up_list
                ave_prob.append(p_e_selected_given_x)
                #t_e_id = um.dh.e2id[um.dh.true_f2e[fr]]
                #print '\tp(e_selected|f)', y_hat[0, e_id]
                #print '\t max p(e|f)', np.max(y_hat)
                #print '\trank e', y_hat_ranks[0, e_id]
                #print '\ty_hat', y_hat[0,:]
            adapted_grad, E_g, E_dx = adapt_grad("adadelta", dW, E_g, E_dx)
            W -= adapted_grad 
            if np.isnan(W).any():
                raise Exception(" updated W has nan")
            prev_user = user
        else:
            pass

    #for u in user2ave_prob:
    #    print u, len(user2ave_prob[u]), np.mean(user2ave_prob[u]), np.std(user2ave_prob[u])
    msg = '\t'.join(['ave over users over steps', str(np.mean(ave_prob)), str(np.std(ave_prob)), str(len(ave_prob))])
    print options
    print msg
