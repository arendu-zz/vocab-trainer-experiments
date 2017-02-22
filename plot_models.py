#!/usr/bin/env python
__author__ = 'arenduchintala'
import pdb
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
            '%d' % int(height),
            ha='center', va='bottom')


if __name__ == '__main__':
    sns.set_style('whitegrid')
    colors = sns.color_palette(n_colors=8).as_hex() #palette="Set2"
    colors.pop(3)
    stats = open('./logs/results/stats.log').read()
    stats = stats.split('done')[0]
    devs = stats.strip().split('DEV mean_losses')
    last_dev = devs.pop()
    tests = last_dev.strip().split('TEST mean_losses')
    first_test = tests.pop(0)
    devs.append(first_test)
    devs = [d for d in devs if d.strip() != '']
    tests = [t for t in tests if t.strip() != '']
    for _set_name, _set in [("dev",devs), ("test",tests)]:
        width = 0.2
        N = 5
        conditions = [] # 5 conditions
        ind = np.arange(N)
        _00_means = [] 
        _00_cis = []
        _10_means = [] 
        _10_cis = []
        _30_means = [] 
        _30_cis = []
        _32_means = [] 
        _32_cis = []
        for d in _set:
            d = d.strip()
            d = d.split('\n')
            condition = d[0]
            _00 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[1]) if i not in [',', '(', ')', ' ', '']]
            _10 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[2]) if i not in [',', '(', ')', ' ', '']]
            _30 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[3]) if i not in [',', '(', ')', ' ', '']]
            _32 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[6]) if i not in [',', '(', ')', ' ', '']]
            conditions.append(condition)
            _00_means.append(_00[1])
            _10_means.append(_10[1])
            _30_means.append(_30[1])
            _32_means.append(_32[1])

            _00_cis.append(0.5 * (_00[4] - _00[3]))
            _10_cis.append(0.5 * (_10[4] - _10[3]))
            _30_cis.append(0.5 * (_30[4] - _30[3]))
            _32_cis.append(0.5 * (_32[4] - _32[3]))

        fig, ax = plt.subplots()
        _00_rects = ax.bar(ind, _00_means, width, color= colors[0], yerr=_00_cis)
        _10_rects = ax.bar(ind + width, _10_means, width, color= colors[1], yerr=_10_cis)
        _30_rects = ax.bar(ind + 2 * width, _30_means, width, color= colors[3], yerr=_30_cis)
        _32_rects = ax.bar(ind + 3 * width, _32_means, width, color= colors[2], yerr=_32_cis)
        ax.set_ylabel('Ave. Loss per user guess')
        ax.set_xticks(ind + (1.5 * width))
        ax.set_xticklabels(tuple(conditions))
        ax.legend((_00_rects[0], _10_rects[0], _30_rects[0], _32_rects[0]), ('m0', 'm1', 'm2', 'm3'))
        #autolabel(_00_rects)
        #autolabel(_10_rects)
        #autolabel(_30_rects)
        #autolabel(_32_rects)
        plt.ylim(0.5,3.2)
        plt.savefig('./images/model-compare-' + _set_name + '.pdf', dpi=200, bbox_inches='tight', pad_inches=0)

