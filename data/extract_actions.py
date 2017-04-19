#!/usr/bin/env python
__author__ = 'arenduchintala'
import sys
import pdb
import numpy as np
import codecs
import argparse
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="accepts a csv file and generates log linear features")
    #insert options here
    opt.add_argument('-v', action='store', dest='vocab_file', default='example', required = True)
    opt.add_argument('-o', action='store', dest='action_file_prefix', required = True)
    options = opt.parse_args()
    data = codecs.open(options.vocab_file, 'r', 'utf8').readlines()
    data = [d.strip().split(',')[1:4] for d in data]
    categories = data[0]
    en2categories = {}
    categories2en = {}
    linenum2en = {}
    linenum2es = {}
    en2linenum = {}
    es2linenum = {}
    test_actions = []
    ex_actions = []
    mc_actions = []
    tp_actions = []
    action_cell = {}
    action_row = {}
    action_col = {}
    pairs2categories = {}
    cat2pairs = {}
    pair2cats = {}
    col2pairs = {}
    row2pairs = {}
    pair2row = {}
    cell2pairs = {}
    test_pairs = []
    train_pairs = []
    for line_idx, line in enumerate(data[1:]):
        for phrase_idx, phrase_pairs in enumerate(line):
            es_str, en_str = phrase_pairs.strip().split('/')
            es_str = es_str.strip()
            en_str = en_str.strip()
            cats = data[0][phrase_idx].split()
            pair = (es_str, en_str)
            ci = col2pairs.get(phrase_idx, [])
            ci.append(pair)
            col2pairs[phrase_idx] = ci

            ri = row2pairs.get(line_idx, [])
            ri.append(pair)
            row2pairs[line_idx] = ri

            pair2row[pair] = line_idx

            cell2pairs[line_idx, phrase_idx] = pair
            for cat in cats:
                cp = cat2pairs.get(cat, [])
                cp.append((es_str, en_str))
                cat2pairs[cat] = cp
            print line_idx, phrase_idx, es_str, en_str, data[0][phrase_idx]

    pdb.set_trace()
    
    test_col_id = 0
    for col_id in col2pairs.keys():
        print col_id, len(col2pairs[col_id])
        test_col_id = test_col_id + 1 if test_col_id < 2 else 0 #np.random.choice(len(col2pairs[col_id]), 1)[0]
        for row_id in xrange(len(col2pairs[col_id])):
            if row_id != test_col_id:
                train_pairs.append(col2pairs[col_id][row_id])
            else:
                test_pairs.append(col2pairs[col_id][test_col_id])

    pdb.set_trace()

    for line_idx, line in enumerate(data[1:]):
        for phrase_idx, phrase_pairs in enumerate(line):
            es_str, en_str = phrase_pairs.strip().split('/')
            es_str = es_str.strip()
            en_str = en_str.strip()
            cats = data[0][phrase_idx].split()
            pair = (es_str, en_str)
            if pair not in test_pairs:
                for cat in cats:
                    cp = cat2pairs.get(cat, [])
                    cp.append((es_str, en_str))
                    cat2pairs[cat] = cp
                    pc = pair2cats.get(pair, [])
                    pc.append(cat)
                    pair2cats[pair] = pc

    pdb.set_trace()
    for train_p in train_pairs:
        a = ('EX', train_p[0], train_p[1])
        ex_actions.append(a)
        a = ('TP', train_p[0], train_p[1])
        tp_actions.append(a)

    for train_p in train_pairs:
        cats = pair2cats[train_p]
        train_p_row = pair2row[train_p]
        has_mc = False
        for cat in cats:
            confusers = [c[1] for c in cat2pairs[cat] if (c != train_p and c not in test_pairs)]
            lemma_confusers = [c[1] for c in cat2pairs[cat] if (pair2row[c] == train_p_row and c != train_p and c not in test_pairs)]
            r_c = list(set(list(np.random.choice(confusers, 4 if len(confusers) >= 4 else len(confusers), replace=False))))
            l_r_c = []
            if len(lemma_confusers)  > 3:
                l_r_c = list(set(list(np.random.choice(lemma_confusers, 4 if len(lemma_confusers) >= 4 else len(lemma_confusers), replace = False))))
            if len(r_c) == 4:
                a =tuple(["MC", train_p[0], train_p[1]] + r_c)
                mc_actions.append(a)
                has_mc = True
            if len(l_r_c) == 4:
                a = tuple(["MC", train_p[0], train_p[1]] + l_r_c)
                mc_actions.append(a)
                has_mc = True
        if not has_mc:
            lemma_confusers = [c[1] for c in row2pairs[train_p_row] if (c not in test_pairs and c != train_p)]
            for _ in xrange(2):
                #throw in 2 mc questions..
                l_r_c = list(set(list(np.random.choice(lemma_confusers, 4 if len(lemma_confusers) >= 4 else len(lemma_confusers), replace = False))))
                if len(l_r_c) == 4:
                    a = tuple(["MC", train_p[0], train_p[1]] + l_r_c)
                    mc_actions.append(a)
                    has_mc = True
                else:
                    pass


    for test_p in test_pairs:
        a = ('TP', test_p[0], test_p[1])
        test_actions.append(a)

    test_file = codecs.open(options.action_file_prefix + '.test', 'w', 'utf8')
    train_file = codecs.open(options.action_file_prefix + '.train', 'w', 'utf8')
    for a_idx, a in enumerate(test_actions):
        test_file.write(str(a_idx) + '|||' + '|||'.join(a) + '\n')
        test_file.flush()
    for a_idx, a in enumerate(ex_actions + mc_actions + tp_actions):
        train_file.write(str(a_idx) + '|||' + '|||'.join(a) + '\n')
        train_file.flush()
    train_file.close()
    test_file.close()
