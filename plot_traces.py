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
=============================================================================
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

best m 0 0
min_loss:            86.7750 57 ./simple.m.m0.u.rms.r.0.001.ur.-1.c.free.bl.0.0.gm.g0.t.t0.log 88.843

best m 1 0
min_loss:            76.7810 46 ./simple.m.m1.u.rms.r.0.01.ur.-1.c.free.bl.0.0.gm.g0.t.t0.log 82.153

best m 3 0
min_loss:            65.2830 9 ./simple.m.m3.u.rms.r.0.001.ur.-1.c.free.bl.0.0.gm.g0.t.t0.log 67.83

best m 0 2
min_loss:            86.7750 56 ./simple.m.m0.u.rms.r.0.001.ur.-1.c.free.bl.0.0.gm.g0.t.t2.log 88.843

best m 1 2
min_loss:            76.5860 37 ./simple.m.m1.u.rms.r.0.001.ur.-1.c.free.bl.0.0.gm.g0.t.t2.log 81.578

best m 3 2
min_loss:            64.8670 7 ./simple.m.m3.u.rms.r.0.01.ur.-1.c.free.bl.0.0.gm.g0.t.t2.log 66.734
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
    colors = sns.color_palette("bright", n_colors=3).as_hex() #palette="Set2"
    d_m2 = ("logs/traces/simple.m.m3.u.rms.r.0.001.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.ut.iter.4", colors[0], "m2")
    d_m1 = ("logs/traces/simple.m.m1.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.ut.iter.13", colors[1], "m1")
    d_m0 = ("logs/traces/simple.m.m0.u.rms.r.0.01.ur.0.0.c.free.bl.0.0.gm.g0.t.t2.ut.iter.50", colors[2], "m0")
    file_data2users = {}
    users2p_data = {}
    for file_path,color,mod in [d_m0, d_m1, d_m2]:
        _file = open(file_path, 'r').read().strip().split('\n')
        for idx in range(0, len(_file), 7):
            _p_y_t = [float(i) for i in _file[idx + 0].split()] 
            _p_y_u = [float(i) for i in _file[idx + 1].split()] 
            _u_c = [int(float(i)) for i in _file[idx + 2].split()] 
            _u_ic = [int(float(i)) for i in _file[idx + 3].split()] 
            _is_mc = [int(float(i)) for i in _file[idx + 4].split()] 
            _is_tp = [int(float(i)) for i in _file[idx + 5].split()] 
            _chance = [float(i) for i in _file[idx + 6].split()] 
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
    for u_idx in users2p_data.keys()[:5]:
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
                    _chance_line.append(_c)
                    _yu = 0.001 if _yu < 0.001 else _yu
                    _ratio = (_yu / _c) if _yu > _c else -1 * (_c/ _yu)
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
            pl, = plt.plot(x_line, _yu_line, lw=1.0, c = color, label=mod)
            lines.append(pl)
            for _s, _x, _y in zip(_yu_markers, x_line, _yu_line):
                plt.plot(_x, _y,marker=_s[0], c=color, markeredgecolor=color, markerfacecolor=_s[1], markersize=8, mew=1.0)

        plt.legend(handles=lines, loc=1)
        #plt.plot(x_line, _chance_line, lw=1.0, ls='-', c = 'black')
        plt.xlim(0, 58)
        plt.show()
