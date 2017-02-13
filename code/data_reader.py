#!/usr/bin/env python
import sys
import codecs
import numpy as np
import theano
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64

def read_data(file_path, dh):
    data_lines = []
    data_lines += codecs.open(file_path, 'r', 'utf8').readlines()
    TRAINING_SEQ = []
    prev_user = None
    X, Y, YT, O, S = None, None, None, None, None
    for line in data_lines:
        user, uts, ptype, tstep, a_idx, fr, en_true, en_options, en_selected, fb, is_qc  = [i.strip() for i in line.split('\t')]
        if user != prev_user:
            if X is not None and Y is not None and O is not None and S is not None and YT is not None:
                TRAINING_SEQ.append((np.array(X, dtype=np.int32),
                    np.array(Y, dtype=np.float32),
                    np.array(YT, dtype=np.int32),
                    np.array(O, dtype=np.float32),
                    np.array(S, dtype=np.float32)))
            else:
                pass
            X, Y, YT, O, S = [],[],[],[],[] #clearing out 
        if en_selected != "NO_ANSWER_MADE":
            x = dh.f2id[fr] #x
            t_id = dh.e2id[en_true]
            e_id = dh.e2id[en_selected] #y index
            y_true = t_id #np.zeros((dh.E_SIZE,)) #y
            #y_true[t_id] = 1.0
            y_selected = np.zeros((dh.E_SIZE,)) #y
            y_selected[e_id] = 1.0 #set what the user selected to 1, and the rest to zero
            if en_options == "ALL":
                o = np.ones((dh.E_SIZE,)).astype(floatX)
            else:
                o = np.zeros((dh.E_SIZE,)).astype(floatX)
                for os in en_options.split(','):
                    o_id = dh.e2id[os.strip()]
                    o[o_id] = 1.0
            o = o.astype(intX)
            y_selected = y_selected.astype(intX)
            t = np.array([0,0,0,0,0,0,0,0,0]).astype(intX)
            t[0] = 1 if ptype in ["EX"] else 0
            t[1] = 1 if ptype in ["TP", "TPR"] else 0
            t[2] = 1 if ptype in ["MC", "MCR"] else 0
            t[3] = 1 if fb == 'revealed' else 0
            t[4] = 1 if fb == 'correct' else 0
            t[5] = 1 if fb == 'incorrect' else 0
            t[6] = 1 if fb == 'nofeedback' else 0
            t[7] = 1 if is_qc == 'test_correct' else 0
            t[8] = 1 if is_qc == 'test_incorrect' else 0

            X.append(x)
            Y.append(y_selected)
            YT.append(y_true)
            O.append(o)
            S.append(t)
        prev_user = user

    TRAINING_SEQ.append((np.array(X, dtype=np.int32),
        np.array(Y, dtype=np.float32),
        np.array(YT, dtype=np.int32),
        np.array(O, dtype=np.float32),
        np.array(S, dtype=np.float32)))
    return TRAINING_SEQ
