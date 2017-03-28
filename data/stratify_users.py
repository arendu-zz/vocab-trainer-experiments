#!/usr/bin/env python
__author__ = 'arenduchintala'
import numpy as np
import sys
import codecs
import argparse
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="takes training data and stratifies based on score")

    #insert options here
    opt.add_argument('-t', action='store', dest='training_data', default='./data_splits/train.data')
    options = opt.parse_args()
    user2scores = {}
    training_data_lines = codecs.open(options.training_data, 'r', 'utf8').readlines()
    for line in training_data_lines:
        items = line.strip().split()
        u = items[0]
        ts = float(items[1])
        ex = 1 if items[2] == 'EX' else 0
        mc = 1 if items[2].startswith('MC') else 0
        tp = 1 if items[2].startswith('TP') else 0
        r = 1 if items[-2] == 'revealed' else 0
        scores = user2scores.get(u, (0,0,0,0,0))
        user2scores[u] = (scores[0] + ex, scores[1] + mc, scores[2] + tp, scores[3] + r, ts)
    user2ratios = {}
    rev2user = []
    for u in user2scores:
        scores = user2scores[u]
        ratios = (scores[0] / scores[-1], scores[1] / scores[-1], scores[2] / scores[-1], scores[3] / scores[-1])
        user2ratios[u] = ratios
        rev2user.append((scores[3] / scores[-1], u))

    ex_ratio = []
    mc_ratio = []
    tp_ratio = []
    r_ratio = []

    for u,s in user2ratios.iteritems():
        ex_ratio.append(s[0])
        mc_ratio.append(s[1])
        tp_ratio.append(s[2])
        r_ratio.append(s[3])

    print 'ex', np.mean(ex_ratio), np.std(ex_ratio)
    print 'mc', np.mean(mc_ratio), np.std(mc_ratio)
    print 'tp', np.mean(tp_ratio), np.std(tp_ratio)
    print 'r', np.mean(r_ratio), np.std(r_ratio) 
    rev2user.sort()
    for r,u in rev2user:
        print u,r, list(user2scores[u])[-1]

    sub_size = 16
    sublists = [rev2user[sub_size * i: sub_size * (i + 1)] for i in range(len(rev2user) / sub_size + 1)]
    strata = {}
    for sub_idx, sub in enumerate(sublists):
        print len(sub), [u for s,u in sub]
        user_set = set([u for s,u in sub])
        f = codecs.open('./data_splits/strata.' +str(sub_idx) + '.data', 'w', 'utf8')
        strata[sub_idx] = (user_set, f)

    for line in training_data_lines:
        items = line.strip().split()
        u = items[0]
        for strata_idx, (user_set, f) in strata.iteritems():
            if u in user_set:
                f.write(line)
                f.flush()
            else:
                pass
    for strata_idx, (user_set, f) in strata.iteritems():
        f.close()
        
    


