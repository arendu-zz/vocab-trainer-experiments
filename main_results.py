#!/usr/bin/env python
import argparse
import sys
import codecs
import numpy as np
import theano
from code.data_reader import read_data
from code.datahelper import DataHelper
from code.simple_loglinear import SimpleLoglinear
from code.eval_tools import disp_eval, eval_losses 
from code.my_utils import load_obj
from scipy import stats as st
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)

__author__ = 'arenduchintala'

if theano.config.floatX == 'float32':
    floatX = np.float32
    intX = np.int32
else:
    floatX = np.float64
    intX = np.int64
"""
best m 0 0
min_loss:            69.6760 17 ./simple.m.m0.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 69.443
max_acc:             0.3510 15 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 0.361
max_pu:              0.2540 15 ./simple.m.m0.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 0.256

best m 1 0
min_loss:            61.3760 16 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 63.274
max_acc:             0.4230 14 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 0.42
max_pu:              0.3220 14 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 0.325

best m 3 0
min_loss:            60.3020 6 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 62.73
max_acc:             0.4310 4 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 0.415
max_pu:              0.3350 8 ./simple.m.m3.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g0.t.t0.log 0.336

best m 0 0
min_loss:            61.0020 15 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 64.563
max_acc:             0.4200 13 ./simple.m.m0.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 0.36
max_pu:              0.2930 15 ./simple.m.m0.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 0.269

best m 1 0
min_loss:            54.0650 16 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 58.203
max_acc:             0.4790 16 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 0.43
max_pu:              0.3480 18 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 0.327

best m 3 0
min_loss:            53.2640 6 ./simple.m.m3.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 57.755
max_acc:             0.4880 8 ./simple.m.m3.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 0.452
max_pu:              0.3700 8 ./simple.m.m3.u.rms.r.0.01.gt.0.c.free.bl.0.0.gm.g1.t.t0.log 0.351
=================================================================================================
all
best m 0 0
min_loss:            99.3360 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 103.892
max_acc:             0.2150 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.214
max_pu:              0.1990 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.193
best m 1 0
min_loss:            59.1320 41 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 62.799
max_acc:             0.4530 41 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.409
max_pu:              0.3060 41 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.286
best m 3 0
min_loss:            55.7200 15 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 59.827
max_acc:             0.4870 15 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.433
max_pu:              0.3480 15 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.334
best m 0 0
min_loss:            84.8320 21 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 83.797
max_acc:             0.2010 21 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.17
max_pu:              0.1600 21 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.162
best m 1 0
min_loss:            66.4370 39 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 67.763
max_acc:             0.3900 39 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.376
max_pu:              0.2770 39 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.279
best m 3 0
min_loss:            62.1550 15 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 65.107
max_acc:             0.4230 15 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.398
max_pu:              0.3280 15 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.328
"""
def ci(a):
    return st.t.interval(0.95, len(a) - 1, loc=np.mean(a), scale=st.sem(a))


def get_mcnemar(acc_list1, acc_list2):
    assert isinstance(acc_list1, list)
    assert isinstance(acc_list2, list)
    t1_pos = np.sum(acc_list1)
    t1_neg = len(acc_list1) - t1_pos
    t2_pos = np.sum(acc_list2)
    t2_neg = len(acc_list2) - t2_pos

    a = [1 if i == j == 1 else 0 for i,j in zip(acc_list1, acc_list2)]
    a = np.sum(a)
    b = [1 if i == 1 and j == 0 else 0 for i,j in zip(acc_list1, acc_list2)]
    b = np.sum(b)
    c = [1 if i == 0 and j == 1 else 0 for i,j in zip(acc_list1, acc_list2)]
    c = np.sum(c)
    d = [1 if i == 0 and j == 0 else 0 for i,j in zip(acc_list1, acc_list2)]
    d = np.sum(d)
    assert a + b == t1_pos
    assert c + d == t1_neg
    assert a + c == t2_pos
    assert b + d == t2_neg
    chisqr = float((b - c) ** 2) / (b + c)
    return a, b, c, d, chisqr


if __name__ == '__main__':
    np.random.seed(124)
    sys.setrecursionlimit(50000)
    opt= argparse.ArgumentParser(description="write program description here")
    opt.add_argument('-f', action='store', dest='feature', default='p.w.pre.suf.c')
    opt.add_argument('--st', action='store', dest='save_trace', default='./', required=True)
    options = opt.parse_args()
    events_file = './data/content/fake-en-medium.' + options.feature  +'.event2feats'
    feats_file = './data/content/fake-en-medium.' + options.feature  +'.feat2id'
    actions_file = './data/content/fake-en-medium.mc.tp.mcr.tpr.actions'
    dh = DataHelper(events_file, feats_file, actions_file)
    TRAINING_SEQ = read_data('./data/data_splits/train.data', dh)
    DEV_SEQ = read_data('./data/data_splits/dev.data', dh)
    TEST_SEQ = read_data('./data/data_splits/test.data', dh)
    m0_g0_sll = load_obj('./logs/models/simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.model')
    m1_g0_sll = load_obj('./logs/models/simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.model')
    m3_g0_sll = load_obj('./logs/models/simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.model')
    m0_g1_sll = load_obj('./logs/models/simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.model')
    m1_g1_sll = load_obj('./logs/models/simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.model')
    m3_g1_sll = load_obj('./logs/models/simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.model')
    for seq_name, SEQ in [("DEV", DEV_SEQ), ("TEST", TEST_SEQ)]:
        l00_ps, l00_ug, l00_ug_c, l00_ug_ic, l00_ug_mc, l00_ug_tp, l00_ug_acc = eval_losses(SEQ, m0_g0_sll, dh)
        l10_ps, l10_ug, l10_ug_c, l10_ug_ic, l10_ug_mc, l10_ug_tp, l10_ug_acc = eval_losses(SEQ, m1_g0_sll, dh)
        l30_ps, l30_ug, l30_ug_c, l30_ug_ic, l30_ug_mc, l30_ug_tp, l30_ug_acc = eval_losses(SEQ, m3_g0_sll, dh)
        l02_ps, l02_ug, l02_ug_c, l02_ug_ic, l02_ug_mc, l02_ug_tp, l02_ug_acc = eval_losses(SEQ, m0_g1_sll, dh)
        l12_ps, l12_ug, l12_ug_c, l12_ug_ic, l12_ug_mc, l12_ug_tp, l12_ug_acc = eval_losses(SEQ, m1_g1_sll, dh)
        l32_ps, l32_ug, l32_ug_c, l32_ug_ic, l32_ug_mc, l32_ug_tp, l32_ug_acc = eval_losses(SEQ, m3_g1_sll, dh)
        assert np.shape(l00_ug) == np.shape(l32_ug)
        assert np.shape(l00_ug_c) == np.shape(l32_ug_c) == np.shape(l12_ug_c)
        assert np.shape(l00_ug_ic) == np.shape(l32_ug_ic) == np.shape(l12_ug_ic)
        assert np.shape(l00_ug_mc) == np.shape(l32_ug_mc) == np.shape(l12_ug_mc)
        assert np.shape(l00_ug_tp) == np.shape(l32_ug_tp) == np.shape(l12_ug_tp)
        #l10_ps, l10_ug = eval_losses(DEV_SEQ, m1_g0_sll, dh)
        #l30_ps, l30_ug = eval_losses(DEV_SEQ, m3_g0_sll, dh)
        #l02_ps, l02_ug = eval_losses(DEV_SEQ, m0_g1_sll, dh)
        #l12_ps, l12_ug = eval_losses(DEV_SEQ, m1_g1_sll, dh)
        #l32_ps, l32_ug = eval_losses(DEV_SEQ, m3_g1_sll, dh)
        print '\n', seq_name, 'mean_losses u'
        print '00', np.mean(l00_ug), np.std(l00_ug), ci(l00_ug)
        print '10', np.mean(l10_ug), np.std(l10_ug), ci(l10_ug)
        print '30', np.mean(l30_ug), np.std(l30_ug), ci(l30_ug)
        print '02', np.mean(l02_ug), np.std(l02_ug), ci(l02_ug)
        print '12', np.mean(l12_ug), np.std(l12_ug), ci(l12_ug)
        print '32', np.mean(l32_ug), np.std(l32_ug), ci(l32_ug)
        print 'pairwise sig tests u'
        print 'ug 00 vs 10:', st.ttest_rel(l00_ug, l10_ug), get_mcnemar(l00_ug_acc, l10_ug_acc)
        print 'ug 10 vs 30:', st.ttest_rel(l10_ug, l30_ug), get_mcnemar(l10_ug_acc, l30_ug_acc)
        print 'ug 02 vs 12:', st.ttest_rel(l02_ug, l12_ug), get_mcnemar(l02_ug_acc, l12_ug_acc)
        print 'ug 12 vs 32:', st.ttest_rel(l12_ug, l32_ug), get_mcnemar(l12_ug_acc, l32_ug_acc)
        print 'ug 00 vs 02:', st.ttest_rel(l00_ug, l02_ug), get_mcnemar(l00_ug_acc, l02_ug_acc)
        print 'ug 10 vs 12:', st.ttest_rel(l10_ug, l12_ug), get_mcnemar(l10_ug_acc, l12_ug_acc)
        print 'ug 30 vs 32:', st.ttest_rel(l30_ug, l32_ug), get_mcnemar(l30_ug_acc, l32_ug_acc)
        print '\n', seq_name, 'mean_losses c'
        print '00', np.mean(l00_ug_c), np.std(l00_ug_c), ci(l00_ug_c)
        print '10', np.mean(l10_ug_c), np.std(l10_ug_c), ci(l10_ug_c)
        print '30', np.mean(l30_ug_c), np.std(l30_ug_c), ci(l30_ug_c)
        print '02', np.mean(l02_ug_c), np.std(l02_ug_c), ci(l02_ug_c)
        print '12', np.mean(l12_ug_c), np.std(l12_ug_c), ci(l12_ug_c)
        print '32', np.mean(l32_ug_c), np.std(l32_ug_c), ci(l32_ug_c)
        print 'pairwise sig tests c'
        print 'ug_c 00 vs 10:', st.ttest_rel(l00_ug_c, l10_ug_c)
        print 'ug_c 10 vs 30:', st.ttest_rel(l10_ug_c, l30_ug_c)
        print 'ug_c 10 vs 12:', st.ttest_rel(l10_ug_c, l12_ug_c)
        print 'ug_c 30 vs 32:', st.ttest_rel(l30_ug_c, l32_ug_c)
        print '\n', seq_name, 'mean_losses ic'
        print '00', np.mean(l00_ug_ic), np.std(l00_ug_ic), ci(l00_ug_ic)
        print '10', np.mean(l10_ug_ic), np.std(l10_ug_ic), ci(l10_ug_ic)
        print '30', np.mean(l30_ug_ic), np.std(l30_ug_ic), ci(l30_ug_ic)
        print '02', np.mean(l02_ug_ic), np.std(l02_ug_ic), ci(l02_ug_ic)
        print '12', np.mean(l12_ug_ic), np.std(l12_ug_ic), ci(l12_ug_ic)
        print '32', np.mean(l32_ug_ic), np.std(l32_ug_ic), ci(l32_ug_ic)
        print 'pairwise sig tests ic'
        print 'ug_ic 00 vs 10:', st.ttest_rel(l00_ug_ic, l10_ug_ic)
        print 'ug_ic 10 vs 30:', st.ttest_rel(l10_ug_ic, l30_ug_ic)
        print 'ug_ic 10 vs 12:', st.ttest_rel(l10_ug_ic, l12_ug_ic)
        print 'ug_ic 30 vs 32:', st.ttest_rel(l30_ug_ic, l32_ug_ic)
        print '\n', seq_name, 'mean_losses mc'
        print '00', np.mean(l00_ug_mc), np.std(l00_ug_mc), ci(l00_ug_mc)
        print '10', np.mean(l10_ug_mc), np.std(l10_ug_mc), ci(l10_ug_mc)
        print '30', np.mean(l30_ug_mc), np.std(l30_ug_mc), ci(l30_ug_mc)
        print '02', np.mean(l02_ug_mc), np.std(l02_ug_mc), ci(l02_ug_mc)
        print '12', np.mean(l12_ug_mc), np.std(l12_ug_mc), ci(l12_ug_mc)
        print '32', np.mean(l32_ug_mc), np.std(l32_ug_mc), ci(l32_ug_mc)
        print 'pairwise sig tests mc'
        print 'ug_mc 00 vs 10:', st.ttest_rel(l00_ug_mc, l10_ug_mc)
        print 'ug_mc 10 vs 30:', st.ttest_rel(l10_ug_mc, l30_ug_mc)
        print 'ug_mc 10 vs 12:', st.ttest_rel(l10_ug_mc, l12_ug_mc)
        print 'ug_mc 30 vs 32:', st.ttest_rel(l30_ug_mc, l32_ug_mc)
        print '\n', seq_name, 'mean_losses tp'
        print '00', np.mean(l00_ug_tp), np.std(l00_ug_tp), ci(l00_ug_tp)
        print '10', np.mean(l10_ug_tp), np.std(l10_ug_tp), ci(l10_ug_tp)
        print '30', np.mean(l30_ug_tp), np.std(l30_ug_tp), ci(l30_ug_tp)
        print '02', np.mean(l02_ug_tp), np.std(l02_ug_tp), ci(l02_ug_tp)
        print '12', np.mean(l12_ug_tp), np.std(l12_ug_tp), ci(l12_ug_tp)
        print '32', np.mean(l32_ug_tp), np.std(l32_ug_tp), ci(l32_ug_tp)
        print 'pairwise sig tests mc'
        print 'ug_tp 00 vs 10:', st.ttest_rel(l00_ug_tp, l10_ug_tp)
        print 'ug_tp 10 vs 30:', st.ttest_rel(l10_ug_tp, l30_ug_tp)
        print 'ug_tp 10 vs 12:', st.ttest_rel(l10_ug_tp, l12_ug_tp)
        print 'ug_mc 30 vs 32:', st.ttest_rel(l30_ug_mc, l32_ug_mc)
    print 'done\n'
    m_names = ["m0_g0", "m1_g0", "m3_g0", "m0_g1", "m1_g1", "m3_g1"]
    models = [m0_g0_sll, m1_g0_sll, m3_g0_sll, m0_g1_sll, m1_g1_sll, m3_g1_sll]
    for m_name, m in zip(m_names, models):
        loaded_msg,loaded_dl,loaded_dpu, loaded_dacc = disp_eval(DEV_SEQ, m, dh, options.save_trace + '.' + m_name + '.dev.traces', 0) 
        print 'loaded dev:', loaded_msg
        loaded_msg,loaded_testl,loaded_testpu, loaded_tacc = disp_eval(TEST_SEQ, m, dh, options.save_trace + '.' + m_name + 'test.traces', 0) 
        print 'loaded test:', loaded_msg
