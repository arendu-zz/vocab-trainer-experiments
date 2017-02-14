#!/usr/bin/env python
import sys
import codecs
import argparse
import matplotlib.pyplot as plt
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'

"""
best m0
min_loss:            25.3430 5 ./simple.m.m0.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log
max_p_u:             0.1560 5 ./simple.m.m0.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log
max_p_c:             0.2120 5 ./simple.m.m0.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log
max_p_ic:            0.1100 5 ./simple.m.m0.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log
max_p_c_diff_ict:    0.0760 5 ./simple.m.m0.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log
min_p_ictt:          0.1070 0 ./simple.m.m0.u.sgd.r.0.1.ur.0.1.c.free.bl.0.5.sl.1.log

best m1
min_loss:            24.9890 95 ./simple.m.m1.u.sgd.r.0.01.ur.0.01.c.free.bl.0.75.sl.1.log
max_p_u:             0.2620 88 ./simple.m.m1.u.sgd.r.0.1.ur.0.01.c.free.bl.0.25.sl.1.log
max_p_c:             0.4350 4 ./simple.m.m1.u.sgd.r.0.01.ur.0.01.c.free.bl.0.25.sl.1.log
max_p_ic:            0.1230 2 ./simple.m.m1.u.sgd.r.0.1.ur.0.01.c.free.bl.0.25.sl.1.log
max_p_c_diff_ict:    0.1560 14 ./simple.m.m1.u.sgd.r.0.1.ur.0.01.c.free.bl.0.25.sl.1.log
min_p_ictt:          0.1170 35 ./simple.m.m1.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log

best m2
min_loss:            25.0400 57 ./simple.m.m2.u.sgd.r.0.01.ur.0.01.c.free.bl.0.75.sl.1.log
max_p_u:             0.2580 34 ./simple.m.m2.u.sgd.r.0.01.ur.0.01.c.free.bl.0.25.sl.1.log
max_p_c:             0.4250 65 ./simple.m.m2.u.sgd.r.0.1.ur.0.01.c.free.bl.0.25.sl.1.log
max_p_ic:            0.1310 75 ./simple.m.m2.u.sgd.r.0.01.ur.0.01.c.free.bl.0.5.sl.1.log
max_p_c_diff_ict:    0.1580 43 ./simple.m.m2.u.sgd.r.0.01.ur.0.1.c.free.bl.0.25.sl.1.log
min_p_ictt:          0.1140 15 ./simple.m.m2.u.sgd.r.0.1.ur.0.1.c.free.bl.0.75.sl.1.log
"""

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")
    #insert options here
    #d_a = "logs/traces/simple.adapt.sgd.0.01.0.1.free.1.ut.iter.1"
    #d_s = "logs/traces/simple.scalar.sgd.0.1.0.01.free.1.ut.iter.1"
    d_m2 = ("logs/traces/simple.m.m2.u.sgd.r.0.01.ur.0.01.c.free.bl.0.75.sl.1.ut.iter.57", "b", "m2")
    d_m1 = ("logs/traces/simple.m.m1.u.sgd.r.0.01.ur.0.01.c.free.bl.0.75.sl.1.ut.iter.95", "g", "m1")
    d_m0 = ("logs/traces/simple.m.m0.u.sgd.r.0.01.ur.0.01.c.free.bl.0.75.sl.1.ut.iter.5", "r", "m0")
    file_data2users = {}
    users2p_data = {}
    for file_path,color,mod in [d_m2, d_m1, d_m0]:
        _file = open(file_path, 'r').read().strip().split('\n')
        for idx in range(0, len(_file), 4):
            _p_y_t = [float(i) for i in _file[idx + 0].split()] 
            _p_y_u = [float(i) for i in _file[idx + 1].split()] 
            _u_c = [int(float(i)) for i in _file[idx + 2].split()] 
            _u_ic = [int(float(i)) for i in _file[idx + 3].split()] 
            p_datas = users2p_data.get(idx, [])
            p_datas.append((file_path, color, mod, _p_y_t, _p_y_u, _u_c, _u_ic))
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

    for u_idx in users2p_data.keys()[:4]:
        lines = []
        for p_data in users2p_data[u_idx]:
            file_path, color, mod, _pyt, _pyu, _uc, _uic = p_data
            _yt_line = []
            _yu_line = []
            _yt_markers = []
            _yu_markers = []
            x_line = []
            for x_idx, (_yt, _yu, _uc, _uic) in enumerate(zip(_pyt, _pyu, _uc, _uic)):
                assert not (_uc == 1 and _uic == 1)
                if _uc  == 1 or _uic == 1:
                    x_line.append(x_idx)
                    if _uc == 1:
                        _yt_markers.append(('o', color))
                        _yt_line.append(_yt)
                        _yu_markers.append(('o', color))
                        _yu_line.append(_yu)
                    elif _uic == 1:
                        _yt_markers.append(('o', 'white'))
                        _yt_line.append(_yt)
                        _yu_markers.append(('o', 'white'))
                        _yu_line.append(_yu)
                    pass
                else:
                    pass
            plt.plot(x_line, _yt_line, lw=0.5, ls=':', c = color)
            pl, = plt.plot(x_line, _yu_line, lw=0.5, c = color, label=mod)
            lines.append(pl)
            for _s, _x, _y in zip(_yu_markers, x_line, _yu_line):
                plt.plot(_x, _y,marker=_s[0], c=color, markeredgecolor=color, markerfacecolor=_s[1], markersize=6.5, mew=0.5)

        plt.legend(handles=lines, loc=3)
        plt.ylim(-0.2, 1.2)
        plt.xlim(0, 58)
        plt.show()
