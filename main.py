#!/usr/bin/env python
from code.loglinear import UserModel
import traceback
import argparse
import sys
import codecs
import numpy as np
import theano
import pdb
#from code.my_utils import cosine_sim
#from scipy.stats.mstats import rankdata as get_ranks

sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64
_eps = np.finfo(np.float32).eps

def create_target(y_selected, y_predicted, feedback):
    #y_selected is a one-hot vector
    #y_predicted is the output based on current set of weights
    #feedback is the indication that the system gives the user i.e. correct or incorrect.
    y_neg_selected = -(y_selected - 1.0)  #flips 1s to 0s and 0s to 1s
    y_predicted = y_predicted * y_neg_selected
    y_predicted = y_predicted / y_predicted.sum(axis=1)[:, np.newaxis]
    y_target = np.where(feedback[:, None], y_selected, y_predicted)
    return y_target

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='feature', default='p.w.pre.suf.c')
    opt.add_argument('-r', action='store', dest='regularization', default=0.01, type=float)
    opt.add_argument('-t', action='store', dest='regularization_type', default='l2')
    opt.add_argument('-l', action='store', dest='learning_rate', default=0.6, type=float)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    #data_lines = codecs.open('./data/data_splits/train.data', 'r', 'utf8').readlines()
    data_lines = []
    data_lines += codecs.open('./data/data_splits/train.data', 'r', 'utf8').readlines()
    reg = options.regularization / float(len(data_lines))
    um = UserModel(events_file, feats_file, actions_file, reg, options.regularization_type)
    user2ave_prob = {}
    ave_prob = []
    user2ave_rank = {}
    prev_user = None
    W = None
    seq_loss = 0.0
    #b = None
    for line in data_lines: 
        user, uts, ptype, tstep, a_idx, fr, en_options, en_selected, fb  = [i.strip() for i in line.split('\t')]
        if user != prev_user:
            sys.stderr.write('.')
            #print b
            print 'seq_loss', seq_loss
            pdb.set_trace()
            #print 'new user', user
            seq_loss = 0.0
            W = np.zeros((um.dh.FEAT_SIZE,)).astype(floatX)
            b = np.zeros((um.dh.E_SIZE,)).astype(floatX)
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
            print 'loss', loss
            seq_loss += loss
            if np.isnan(loss):
                raise Exception("loss is nan")
            dW = um.get_grad(W, x, o, y_selected, f)
            dW[np.absolute(dW) < _eps] = 0.0
            #TODO: is this gradient ok in theano??
            if np.isnan(dW).any(): 
                raise Exception("grad W has nan")
            #if np.isnan(db).any():
            #    raise Exception("grad b has nan")
            y_hat = um.y_given_x(W, x, o)
            if np.isnan(y_hat).any():
                raise Exception("y_hat has nan")
            if ptype != 'EX' and fb != 'revealed':
                p_e_selected_given_x = y_hat[0, e_id]
                up_list = user2ave_prob.get(user, [])
                up_list.append(p_e_selected_given_x)
                user2ave_prob[user] = up_list
                ave_prob.append(p_e_selected_given_x)
            else:
                pass # don't include in eval metric..
            try:
                if ptype == "EX":
                    W -= options.learning_rate * dW
                    #b -= options.learning_rate * db
                elif ptype == "MC" or ptype == "MCR":
                    W -= options.learning_rate * dW
                    #b -= options.learning_rate * db
                elif ptype == "TP" or ptype == "TPR":
                    W -= options.learning_rate * dW
                    #b -= options.learning_rate * db
                else:
                    raise Exception("unknown ptype")
            except: 
                print traceback.print_exc()
                pdb.set_trace()

            prev_user = user
        else:
            pass

    #for u in user2ave_prob:
    #    print u, len(user2ave_prob[u]), np.mean(user2ave_prob[u]), np.std(user2ave_prob[u])
    msg = '\t'.join(['ave over users over steps', str(np.mean(ave_prob)), str(np.std(ave_prob)), str(len(ave_prob))])
    print options
    print msg
