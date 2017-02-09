#!/usr/bin/env python
import sys
import codecs
import pdb
import argparse
import matplotlib.pyplot as plt
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'
"""
min_loss:            10.1440 0 ./simple.scalar.sgd.0.0.0.01.free.1.log
max_p_u:             0.1320 8 ./simple.scalar.sgd.0.1.0.01.free.1.log
max_p_c:             0.1610 1 ./simple.scalar.sgd.0.1.0.01.free.1.log
max_p_ic:            0.1070 0 ./simple.scalar.sgd.0.0.0.01.free.1.log
max_p_c_diff_ict:    0.0540 1 ./simple.scalar.sgd.0.1.0.01.free.1.log
min_p_ictt:          0.1070 0 ./simple.scalar.sgd.0.0.0.01.free.1.log

best adapt
min_loss:            11.6290 48 ./simple.adapt.sgd.0.01.0.01.free.1.log
max_p_u:             0.2390 0 ./simple.adapt.sgd.0.0.0.01.free.1.log
max_p_c:             0.3700 0 ./simple.adapt.sgd.0.0.0.01.free.1.log
max_p_ic:            0.1350 4 ./simple.adapt.sgd.0.01.0.01.free.1.log
max_p_c_diff_ict:    0.1130 1 ./simple.adapt.sgd.0.01.0.1.free.1.log
min_p_ictt:          0.1050 32 ./simple.adapt.sgd.0.01.0.5.free.1.log
"""

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    d_a = "logs/traces/simple.adapt.sgd.0.01.0.1.free.1.ut.iter.1"
    d_s = "logs/traces/simple.scalar.sgd.0.1.0.01.free.1.ut.iter.1"
    opt.add_argument('-t', action='store', dest='adapt_trace_file', default=d_a)
    opt.add_argument('-c', action='store', dest='scalar_trace_file', default=d_s)
    options = opt.parse_args()
    adapt_trace_file = open(options.adapt_trace_file, 'r').read().strip().split('\n')
    scalar_trace_file = open(options.scalar_trace_file, 'r').read().strip().split('\n')
    users = {}
    for idx in range(0, len(adapt_trace_file), 4):
        adapt_p_y_t = [float(i) for i in adapt_trace_file[idx + 0].split()] 
        adapt_p_y_u = [float(i) for i in adapt_trace_file[idx + 1].split()] 
        adapt_u_c = [int(float(i)) for i in adapt_trace_file[idx + 2].split()] 
        adapt_u_ic = [int(float(i)) for i in adapt_trace_file[idx + 3].split()] 
        scalar_p_y_t = [float(i) for i in scalar_trace_file[idx + 0].split()] 
        scalar_p_y_u = [float(i) for i in scalar_trace_file[idx + 1].split()] 
        scalar_u_c = [int(float(i)) for i in scalar_trace_file[idx + 2].split()] 
        scalar_u_ic = [int(float(i)) for i in scalar_trace_file[idx + 3].split()] 
        users[len(users)] = (adapt_p_y_t, adapt_p_y_u, adapt_u_c, adapt_u_ic, scalar_p_y_t, scalar_p_y_u, scalar_u_c, scalar_u_ic)

    for u_idx in [0,1,2,3,4,5,6,7,8,9,10]:
        adapt_pyt, adapt_pyu, adapt_uc, adapt_uic, scalar_pyt, scalar_pyu, scalar_uc, scalar_uic = users[u_idx]
        adapt_yt_line = []
        adapt_yu_line = []
        adapt_yt_markers = []
        adapt_yu_markers = []
        scalar_yt_line = []
        scalar_yu_line = []
        scalar_yt_markers = []
        scalar_yu_markers = []
        x_line = []
        adapt_c = 'b'
        for x_idx, (_yt, _yu, _uc, _uic) in enumerate(zip(adapt_pyt, adapt_pyu, adapt_uc, adapt_uic)):
            assert not (_uc == 1 and _uic == 1)
            if _uc  == 1 or _uic == 1:
                x_line.append(x_idx)
                if _uc == 1:
                    adapt_yt_markers.append(('o', adapt_c))
                    adapt_yt_line.append(_yt)
                    adapt_yu_markers.append(('o', adapt_c))
                    adapt_yu_line.append(_yu)
                elif _uic == 1:
                    adapt_yt_markers.append(('o', 'white'))
                    adapt_yt_line.append(_yt)
                    adapt_yu_markers.append(('o', 'white'))
                    adapt_yu_line.append(_yu)
                pass
            else:
                pass
        plt.plot(x_line, adapt_yt_line, lw=0.5, ls=':', c = adapt_c,  label='s_pt')
        plt.plot(x_line, adapt_yu_line, lw=0.5, c = adapt_c, label='s_pu')
        for _s, _x, _y in zip(adapt_yu_markers, x_line, adapt_yu_line):
            plt.plot(_x, _y,marker=_s[0], c=adapt_c, markeredgecolor=adapt_c, markerfacecolor=_s[1], markersize=6.5, mew=0.5)

        x_line = []
        scalar_c = 'g'
        for x_idx, (_yt, _yu, _uc, _uic) in enumerate(zip(scalar_pyt, scalar_pyu, scalar_uc, scalar_uic)):
            assert not (_uc == 1 and _uic == 1)
            if _uc  == 1 or _uic == 1:
                x_line.append(x_idx)
                if _uc == 1:
                    scalar_yt_markers.append(('o', scalar_c))
                    scalar_yt_line.append(_yt)
                    scalar_yu_markers.append(('o', scalar_c))
                    scalar_yu_line.append(_yu)
                elif _uic == 1:
                    scalar_yt_markers.append(('o', 'white'))
                    scalar_yt_line.append(_yt)
                    scalar_yu_markers.append(('o', 'white'))
                    scalar_yu_line.append(_yu)
                pass
            else:
                pass
        plt.plot(x_line, scalar_yt_line,lw=0.5, ls=':', c = scalar_c,  label='s_pt')
        plt.plot(x_line, scalar_yu_line,lw=0.5, c = scalar_c, label='s_pu')
        for _s, _x, _y in zip(scalar_yu_markers, x_line, scalar_yu_line):
            plt.plot(_x, _y,marker=_s[0], c=scalar_c, markeredgecolor=scalar_c, markerfacecolor=_s[1], markersize=6.5, mew=0.5)
        plt.ylim(-0.2, 1.2)
        plt.xlim(0, 58)
        plt.show()
        #plt.savefig('temp.' + str(u_idx) + '.png', dpi=200)
