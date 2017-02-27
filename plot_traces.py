#!/usr/bin/env python
import sys
from scipy import stats
import codecs
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'

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
==================================================================================================
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
============================================================================================================
best m 0 0
min_loss:            117.2730 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 116.321
max_acc:             0.1810 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.199
max_pu:              0.1690 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.184
best m 1 0
min_loss:            70.4900 70 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 73.941
max_acc:             0.4370 70 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.409
max_pu:              0.2730 70 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.269
best m 3 0
min_loss:            58.8880 34 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 63.877
max_acc:             0.4720 34 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.445
max_pu:              0.3450 34 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.log 0.327
best m 0 0
min_loss:            90.6160 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 88.863
max_acc:             0.1820 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.158
max_pu:              0.1560 5 ./simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.16
best m 1 0
min_loss:            65.1110 40 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 66.754
max_acc:             0.3970 40 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.389
max_pu:              0.2850 40 ./simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.285
best m 3 0
min_loss:            62.2900 14 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 65.38
max_acc:             0.4270 14 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.385
max_pu:              0.3290 14 ./simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g0.t.t0.top_all.log 0.327
"""


if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    #d_a = "logs/traces/simple.adapt.sgd.0.01.0.1.free.1.ut.iter.1"
    #d_s = "logs/traces/simple.scalar.sgd.0.1.0.01.free.1.ut.iter.1"
    #print sns.palplot(sns.color_palette("Blues"))
    #flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    sns.set_style("whitegrid")
    #colors = sns.color_palette("Set2", n_colors=3).as_hex() #palette="Set2"
    #colors = sns.color_palette(n_colors=8).as_hex() #palette="Set2"
    colors = sns.color_palette("Paired",n_colors=6).as_hex() #palette="Set2"
    #colors.pop(3)
    d_m2 = ("logs/traces/simple.m.m3.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.ut.iter.34", colors[5], "cm")
    d_m1 = ("logs/traces/simple.m.m1.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.ut.iter.70", colors[3], "vm")
    d_m0 = ("logs/traces/simple.m.m0.u.rms.r.0.001.gt.0.c.free.bl.0.0.gm.g1.t.t0.top_all.ut.iter.5", colors[1], "sm")
    #d_m2 = ("logs/results/traces.m3_t2.dev.traces.iter.0",colors[2], "m3")
    #d_m1 = ("logs/results/traces.m1_t0.dev.traces.iter.0", colors[1], "m1")
    #d_m0 = ("logs/results/traces.m0_t0.dev.traces.iter.0", colors[0], "m0")
    file_data2users = {}
    users2p_data = {}
    for file_path,color,mod in [d_m0, d_m2]:
        _file = open(file_path, 'r').read().strip().split('\n')
        for idx in range(0, len(_file), 8):
            _p_y_t = [float(i) for i in _file[idx + 0].split()] 
            _p_y_u = [float(i) for i in _file[idx + 1].split()] 
            _u_c = [int(float(i)) for i in _file[idx + 2].split()] 
            _u_ic = [int(float(i)) for i in _file[idx + 3].split()] 
            _is_mc = [int(float(i)) for i in _file[idx + 4].split()] 
            _is_tp = [int(float(i)) for i in _file[idx + 5].split()] 
            _chance = [float(i) for i in _file[idx + 6].split()] 
            if 0.0 in _chance:
                print file_path, idx + 6
                print _chance
                raise BaseException("zero in _chance")
            p_datas = users2p_data.get(idx, [])
            p_datas.append((file_path, color, mod, _p_y_t, _p_y_u, _u_c, _u_ic, _is_mc, _is_tp, _chance))
            users2p_data[idx] = p_datas

        pass
    #adapt_trace_file = open(options.adapt_trace_file, 'r').read().strip().split('\n')
    #scalar_trace_file = open(options.scalar_trace_file, 'r').read().strip().split('\n')
    #users = {}
    #for idx in range(0, len(adapt_trace_file), 4):
    #    adapt_p_y_t = [float(i) for i in adapt_trace_file[idx + 0].split()] 
    #    adapt_p_y_u = [float(i) for i in adapt_trace_file[idx + 1].split()] 
    #    adapt_u_c = [int(float(i)) for i in adapt_trace_file[idx + 2].split()] 
    #    adapt_u_ic = [int(float(i)) for i in adapt_trace_file[idx + 3].split()] 
    #    scalar_p_y_t = [float(i) for i in scalar_trace_file[idx + 0].split()] 
    #    scalar_p_y_u = [float(i) for i in scalar_trace_file[idx + 1].split()] 
    #    scalar_u_c = [int(float(i)) for i in scalar_trace_file[idx + 2].split()] 
    #    scalar_u_ic = [int(float(i)) for i in scalar_trace_file[idx + 3].split()] 
    #    users[len(users)] = (adapt_p_y_t, adapt_p_y_u, adapt_u_c, adapt_u_ic, scalar_p_y_t, scalar_p_y_u, scalar_u_c, scalar_u_ic)
    for u_idx in users2p_data.keys()[:]:
        lines = []
        for p_data in users2p_data[u_idx]:
            file_path, color, mod, _pyt, _pyu, _uc, _uic, _is_mc, _is_tp, _chance = p_data
            _yt_line = []
            _yu_line = []
            _yt_markers = []
            _yu_markers = []
            x_line = []
            _chance_line = []
            for x_idx, (_yt, _yu, _uc, _uic, _mc, _tp, _c) in enumerate(zip(_pyt, _pyu, _uc, _uic, _is_mc, _is_tp, _chance)):
                assert not (_uc == 1 and _uic == 1)
                if _uc  == 1 or _uic == 1:
                    x_line.append(x_idx)
                    _chance_line.append(1.0)
                    _yu = 0.001 if _yu < 0.001 else _yu
                    _ratio = (_yu / _c) #if _yu > _c else -1 * (_c/ _yu)
                    _yu_line.append(_ratio)
                    _yt_line.append(_yt)
                    if _uc == 1:
                        _yt_markers.append(('o' if _mc == 1 else 's', color))
                        _yu_markers.append(('o' if _mc == 1 else 's', color))
                    elif _uic == 1:
                        _yt_markers.append(('o' if _mc == 1 else 's', 'white'))
                        _yu_markers.append(('o' if _mc == 1 else 's', 'white'))
                    pass
                else:
                    pass
            #plt.plot(x_line, _yt_line, lw=0.5, ls=':', c = color)
            min_x_line = min(x_line)
            x_line = [x_ - min_x_line for x_ in x_line]
            pl, = plt.plot(x_line, _yu_line, lw=2.0, c = color, label=mod)
            lines.append(pl)
            for _s, _x, _y in zip(_yu_markers, x_line, _yu_line):
                plt.plot(_x, _y,marker=_s[0], c=color, markeredgecolor=color, markerfacecolor=_s[1], markersize=7, mew=2.0)

        plt.legend(handles=lines, loc=2)
        plt.plot(x_line, _chance_line, lw=1.0, ls='-', c = 'black')
        #plt.show()
        plt.xlim(xmin=0)
        plt.yscale("log")
        plt.ylabel("P(u) over chance")
        plt.xlabel("training steps")
        plt.title("Model Comparison (single user)")
        plt.savefig("./images/fig" + str(u_idx) + ".pdf", dpi=200,bbox_inches='tight', pad_inches=0)
        plt.gcf().clear()

