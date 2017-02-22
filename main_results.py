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
from scipy import stats
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
min_loss:            72.5520 34 ./simple.m.m0.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t0.log 76.272

best m 1 0
min_loss:            66.2700 29 ./simple.m.m1.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t0.log 68.335

best m 3 0
min_loss:            65.1620 4 ./simple.m.m3.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t0.log 67.084

best m 0 2
min_loss:            71.6510 50 ./simple.m.m0.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.log 75.798

best m 1 2
min_loss:            65.7450 13 ./simple.m.m1.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.log 68.3

best m 3 2
min_loss:            64.2370 4 ./simple.m.m3.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.log 66.145
"""
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
    m0_t0_sll = load_obj('./logs/models/simple.m.m0.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t0.model')
    m1_t0_sll = load_obj('./logs/models/simple.m.m1.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t0.model')
    m3_t0_sll = load_obj('./logs/models/simple.m.m3.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t0.model')
    m0_t2_sll = load_obj('./logs/models/simple.m.m0.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.model')
    m1_t2_sll = load_obj('./logs/models/simple.m.m1.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.model')
    m3_t2_sll = load_obj('./logs/models/simple.m.m3.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.model')
    for seq_name, SEQ in [("DEV", DEV_SEQ), ("TEST", TEST_SEQ)]:
        l00_ps, l00_ug, l00_ug_c, l00_ug_ic, l00_ug_mc, l00_ug_tp = eval_losses(SEQ, m0_t0_sll, dh)
        l10_ps, l10_ug, l10_ug_c, l10_ug_ic, l10_ug_mc, l10_ug_tp = eval_losses(SEQ, m1_t0_sll, dh)
        l30_ps, l30_ug, l30_ug_c, l30_ug_ic, l30_ug_mc, l30_ug_tp = eval_losses(SEQ, m3_t0_sll, dh)
        l02_ps, l02_ug, l02_ug_c, l02_ug_ic, l02_ug_mc, l02_ug_tp = eval_losses(SEQ, m0_t2_sll, dh)
        l12_ps, l12_ug, l12_ug_c, l12_ug_ic, l12_ug_mc, l12_ug_tp = eval_losses(SEQ, m1_t2_sll, dh)
        l32_ps, l32_ug, l32_ug_c, l32_ug_ic, l32_ug_mc, l32_ug_tp = eval_losses(SEQ, m3_t2_sll, dh)
        assert np.shape(l00_ug) == np.shape(l32_ug)
        assert np.shape(l00_ug_c) == np.shape(l32_ug_c) == np.shape(l12_ug_c)
        assert np.shape(l00_ug_ic) == np.shape(l32_ug_ic) == np.shape(l12_ug_ic)
        assert np.shape(l00_ug_mc) == np.shape(l32_ug_mc) == np.shape(l12_ug_mc)
        assert np.shape(l00_ug_tp) == np.shape(l32_ug_tp) == np.shape(l12_ug_tp)
        #l10_ps, l10_ug = eval_losses(DEV_SEQ, m1_t0_sll, dh)
        #l30_ps, l30_ug = eval_losses(DEV_SEQ, m3_t0_sll, dh)
        #l02_ps, l02_ug = eval_losses(DEV_SEQ, m0_t2_sll, dh)
        #l12_ps, l12_ug = eval_losses(DEV_SEQ, m1_t2_sll, dh)
        #l32_ps, l32_ug = eval_losses(DEV_SEQ, m3_t2_sll, dh)
        print '\n', seq_name, 'mean_losses u'
        print '00', np.mean(l00_ug), np.std(l00_ug)
        print '10', np.mean(l10_ug), np.std(l10_ug)
        print '30', np.mean(l30_ug), np.std(l30_ug)
        print '02', np.mean(l02_ug), np.std(l02_ug)
        print '12', np.mean(l12_ug), np.std(l12_ug)
        print '32', np.mean(l32_ug), np.std(l32_ug)
        print 'pairwise sig tests u'
        print 'ug 00 vs 10:', stats.ttest_rel(l00_ug, l10_ug)
        print 'ug 10 vs 30:', stats.ttest_rel(l10_ug, l30_ug)
        print 'ug 10 vs 12:', stats.ttest_rel(l10_ug, l12_ug)
        print 'ug 30 vs 32:', stats.ttest_rel(l30_ug, l32_ug)
        print '\n', seq_name, 'mean_losses c'
        print '00', np.mean(l00_ug_c), np.std(l00_ug_c)
        print '10', np.mean(l10_ug_c), np.std(l10_ug_c)
        print '30', np.mean(l30_ug_c), np.std(l30_ug_c)
        print '02', np.mean(l02_ug_c), np.std(l02_ug_c)
        print '12', np.mean(l12_ug_c), np.std(l12_ug_c)
        print '32', np.mean(l32_ug_c), np.std(l32_ug_c)
        print 'pairwise sig tests c'
        print 'ug_c 00 vs 10:', stats.ttest_rel(l00_ug_c, l10_ug_c)
        print 'ug_c 10 vs 30:', stats.ttest_rel(l10_ug_c, l30_ug_c)
        print 'ug_c 10 vs 12:', stats.ttest_rel(l10_ug_c, l12_ug_c)
        print 'ug_c 30 vs 32:', stats.ttest_rel(l30_ug_c, l32_ug_c)
        print '\n', seq_name, 'mean_losses ic'
        print '00', np.mean(l00_ug_ic), np.std(l00_ug_ic)
        print '10', np.mean(l10_ug_ic), np.std(l10_ug_ic)
        print '30', np.mean(l30_ug_ic), np.std(l30_ug_ic)
        print '02', np.mean(l02_ug_ic), np.std(l02_ug_ic)
        print '12', np.mean(l12_ug_ic), np.std(l12_ug_ic)
        print '32', np.mean(l32_ug_ic), np.std(l32_ug_ic)
        print 'pairwise sig tests ic'
        print 'ug_ic 00 vs 10:', stats.ttest_rel(l00_ug_ic, l10_ug_ic)
        print 'ug_ic 10 vs 30:', stats.ttest_rel(l10_ug_ic, l30_ug_ic)
        print 'ug_ic 10 vs 12:', stats.ttest_rel(l10_ug_ic, l12_ug_ic)
        print 'ug_ic 30 vs 32:', stats.ttest_rel(l30_ug_ic, l32_ug_ic)
        print '\n', seq_name, 'mean_losses mc'
        print '00', np.mean(l00_ug_mc), np.std(l00_ug_mc)
        print '10', np.mean(l10_ug_mc), np.std(l10_ug_mc)
        print '30', np.mean(l30_ug_mc), np.std(l30_ug_mc)
        print '02', np.mean(l02_ug_mc), np.std(l02_ug_mc)
        print '12', np.mean(l12_ug_mc), np.std(l12_ug_mc)
        print '32', np.mean(l32_ug_mc), np.std(l32_ug_mc)
        print 'pairwise sig tests mc'
        print 'ug_mc 00 vs 10:', stats.ttest_rel(l00_ug_mc, l10_ug_mc)
        print 'ug_mc 10 vs 30:', stats.ttest_rel(l10_ug_mc, l30_ug_mc)
        print 'ug_mc 10 vs 12:', stats.ttest_rel(l10_ug_mc, l12_ug_mc)
        print 'ug_mc 30 vs 32:', stats.ttest_rel(l30_ug_mc, l32_ug_mc)
        print '\n', seq_name, 'mean_losses tp'
        print '00', np.mean(l00_ug_tp), np.std(l00_ug_tp)
        print '10', np.mean(l10_ug_tp), np.std(l10_ug_tp)
        print '30', np.mean(l30_ug_tp), np.std(l30_ug_tp)
        print '02', np.mean(l02_ug_tp), np.std(l02_ug_tp)
        print '12', np.mean(l12_ug_tp), np.std(l12_ug_tp)
        print '32', np.mean(l32_ug_tp), np.std(l32_ug_tp)
        print 'pairwise sig tests mc'
        print 'ug_tp 00 vs 10:', stats.ttest_rel(l00_ug_tp, l10_ug_tp)
        print 'ug_tp 10 vs 30:', stats.ttest_rel(l10_ug_tp, l30_ug_tp)
        print 'ug_tp 10 vs 12:', stats.ttest_rel(l10_ug_tp, l12_ug_tp)
        print 'ug_mc 30 vs 32:', stats.ttest_rel(l30_ug_mc, l32_ug_mc)
    print 'done\n'
    m_names = ["m0_t0", "m1_t0", "m3_t0", "m0_t2", "m1_t2", "m3_t2"]
    models = [m0_t0_sll, m1_t0_sll, m3_t0_sll, m0_t2_sll, m1_t2_sll, m3_t2_sll]
    for m_name, m in zip(m_names, models):
        loaded_msg,loaded_dl,loaded_dpu = disp_eval(DEV_SEQ, m, dh, options.save_trace + '.' + m_name + '.dev.traces', 0) 
        print 'loaded dev:', loaded_msg
        loaded_msg,loaded_testl,loaded_testpu = disp_eval(TEST_SEQ, m, dh, options.save_trace + '.' + m_name + 'test.traces', 0) 
        print 'loaded test:', loaded_msg
