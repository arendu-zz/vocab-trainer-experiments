#!/usr/bin/env python
import sys
import codecs
import pdb
import argparse
import matplotlib.pyplot as plt
sys.stdout = codecs.getwriter('utf-8')(sys.stdout)
__author__ = 'arenduchintala'
"""
best scalar
min_loss: (4.007, 2, './simple.scalar.sgd.0.0.0.01.free.log')
max_p_r: (0.249, 46, './simple.scalar.sgd.0.001.0.0.free.log')
max_p_u: (0.279, 3, './simple.scalar.sgd.0.001.0.01.free.log')
max_p_c: (0.473, 46, './simple.scalar.sgd.0.001.0.0.free.log')
max_p_ic: (0.149, 46, './simple.scalar.sgd.0.001.0.0.free.log')
best adapt
min_loss: (4.035, 48, './simple.adapt.sgd.0.0.0.01.free.log')
max_p_r: (0.407, 2, './simple.adapt.sgd.0.1.0.0.free.log')
max_p_u: (0.339, 9, './simple.adapt.sgd.0.01.0.01.free.log')
max_p_c: (0.64, 12, './simple.adapt.sgd.0.0.0.0.free.log')
max_p_ic: (0.185, 28, './simple.adapt.sgd.0.001.0.0.free.log')
"""

if __name__ == '__main__':
    opt= argparse.ArgumentParser(description="write program description here")

    #insert options here
    d_a = "logs/traces/simple.adapt.sgd.0.001.0.01.free.ut.iter.28"
    d_s = "logs/traces/simple.scalar.sgd.0.001.0.01.free.ut.iter.9"
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
    for u_idx in [0,1,2]:
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
        for x_idx, (_yt, _yu, _uc, _uic) in enumerate(zip(adapt_pyt, adapt_pyu, adapt_uc, adapt_uic)):
            assert not (_uc == 1 and _uic == 1)
            if _uc  == 1 or _uic == 1:
                x_line.append(x_idx)
                if _uc == 1:
                    adapt_yt_markers.append('.')
                    adapt_yt_line.append(_yt)
                    adapt_yu_markers.append('o')
                    adapt_yu_line.append(_yu)
                elif _uic == 1:
                    adapt_yt_markers.append('.')
                    adapt_yt_line.append(_yt)
                    adapt_yu_markers.append('x')
                    adapt_yu_line.append(_yu)
                pass
            else:
                pass
        plt.plot(x_line, adapt_yt_line,c = 'b', alpha=0.5, label='v_pt', linestyle=':')
        plt.plot(x_line, adapt_yu_line,c = 'b', label='v_pu')
        for _s, _x, _y in zip(adapt_yu_markers, x_line, adapt_yu_line):
            plt.scatter(_x, _y, marker=_s, c='b')

        x_line = []
        for x_idx, (_yt, _yu, _uc, _uic) in enumerate(zip(scalar_pyt, scalar_pyu, scalar_uc, scalar_uic)):
            assert not (_uc == 1 and _uic == 1)
            if _uc  == 1 or _uic == 1:
                x_line.append(x_idx)
                if _uc == 1:
                    scalar_yt_markers.append('.')
                    scalar_yt_line.append(_yt)
                    scalar_yu_markers.append('o')
                    scalar_yu_line.append(_yu)
                elif _uic == 1:
                    scalar_yt_markers.append('.')
                    scalar_yt_line.append(_yt)
                    scalar_yu_markers.append('x')
                    scalar_yu_line.append(_yu)
                pass
            else:
                pass
        plt.plot(x_line, scalar_yt_line,c = 'r', alpha=0.5, label='s_pt', linestyle=':')
        plt.plot(x_line, scalar_yu_line,c = 'r', label='s_pu')
        for _s, _x, _y in zip(scalar_yu_markers, x_line, scalar_yu_line):
            plt.scatter(_x, _y, marker=_s, c='r')
        plt.ylim(-0.2, 1.2)
        plt.xlim(0, 58)
        plt.show()
        #plt.savefig('temp.png', dpi=200)
