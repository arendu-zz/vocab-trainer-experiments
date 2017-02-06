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
    X, Y, O, F, S = None, None, None, None, None
    for line in data_lines:
        user, uts, ptype, tstep, a_idx, fr, en_options, en_selected, fb  = [i.strip() for i in line.split('\t')]
        if user != prev_user:
            if X is not None and Y is not None and O is not None and F is not None and S is not None:
                TRAINING_SEQ.append((np.array(X, dtype=np.int32),
                    np.array(Y, dtype=np.int32),
                    np.array(O, dtype=np.int32),
                    np.array(F, dtype=np.int32),
                    np.array(S, dtype=np.int32)))
            else:
                pass
            X, Y, O, F, S = [],[],[],[],[] #clearing out 
        if en_selected != "NO_ANSWER_MADE":
            x = dh.f2id[fr] #x
            e_id = dh.e2id[en_selected] #y index
            y_selected = np.zeros((dh.E_SIZE,)) #y
            y_selected[e_id] = 1.0 #set what the user selected to 1, and the rest to zero
            if en_options == "ALL":
                o = np.ones((dh.E_SIZE,)).astype(floatX)
            else:
                o = np.zeros((dh.E_SIZE,)).astype(floatX)
                for os in en_options.split(','):
                    o_id = dh.e2id[os.strip()]
                    o[o_id] = 1.0
            f = 0 if fb == 'incorrect' else 1 #both corrent and reveal go to 1
            o = o.astype(intX)
            y_selected = y_selected.astype(intX)
            t = np.array([0,0,0,0,0,0]).astype(intX)
            t[3] = 1 if fb == 'revealed' else 0
            t[4] = 1 if fb == 'correct' else 0
            t[5] = 1 if fb == 'incorrect' else 0
            if ptype == "EX":
                t[0] = 1
            elif ptype in ["MC", "MCR"]:
                t[1] = 1
            elif ptype in ["TP", "TPR"]:
                t[2] = 1
            else:
                raise Exception("unknown prompt type:" + ptype)

            X.append(x)
            Y.append(y_selected)
            O.append(o)
            F.append(f)
            S.append(t)
        prev_user = user

    TRAINING_SEQ.append((np.array(X, dtype=np.int32),
        np.array(Y, dtype=np.int32),
        np.array(O, dtype=np.int32),
        np.array(F, dtype=np.int32),
        np.array(S, dtype=np.int32)))
    return TRAINING_SEQ
