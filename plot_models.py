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
    colors = sns.color_palette("Paired",n_colors=6).as_hex() #palette="Set2"
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
        for plt_type in "acc loss".split():
            width = 0.1
            N = 5
            conditions = [] # 5 conditions
            ind = np.arange(N)
            _00_means = [] 
            _00_cis = []
            _10_means = [] 
            _10_cis = []
            _30_means = [] 
            _30_cis = []
            _02_means = [] 
            _02_cis = []
            _12_means = [] 
            _12_cis = []
            _32_means = [] 
            _32_cis = []
            _00_acc = []
            _10_acc = []
            _30_acc = []
            _02_acc = []
            _12_acc = []
            _32_acc = []
            for d in _set:
                d = d.strip()
                d = d.split('\n')
                condition = d[0]
                _00 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[1]) if i not in [',', '(', ')', ' ', '']]
                _10 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[2]) if i not in [',', '(', ')', ' ', '']]
                _30 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[3]) if i not in [',', '(', ')', ' ', '']]
                _02 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[4]) if i not in [',', '(', ')', ' ', '']]
                _12 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[5]) if i not in [',', '(', ')', ' ', '']]
                _32 = [float(i) for i in re.split(r'(\s|,|\(|\))', d[6]) if i not in [',', '(', ')', ' ', '']]
                conditions.append(condition)
                _00_means.append(_00[1])
                _10_means.append(_10[1])
                _30_means.append(_30[1])
                _02_means.append(_02[1])
                _12_means.append(_12[1])
                _32_means.append(_32[1])

                _00_cis.append(0.5 * (_00[4] - _00[3]))
                _10_cis.append(0.5 * (_10[4] - _10[3]))
                _30_cis.append(0.5 * (_30[4] - _30[3]))
                _02_cis.append(0.5 * (_02[4] - _02[3]))
                _12_cis.append(0.5 * (_12[4] - _12[3]))
                _32_cis.append(0.5 * (_32[4] - _32[3]))

                _00_acc.append(_00[-1])
                _10_acc.append(_10[-1])
                _30_acc.append(_30[-1])
                _02_acc.append(_02[-1])
                _12_acc.append(_12[-1])
                _32_acc.append(_32[-1])


            fig, ax = plt.subplots()
            if plt_type == "loss":
                plt.ylim(0.5,4.4)
                _00_rects = ax.bar(ind, _00_means, width, color= colors[0], yerr=_00_cis)
                _02_rects = ax.bar(ind + width, _02_means, width, color= colors[1], yerr=_02_cis)
                _10_rects = ax.bar(ind + 2 * width, _10_means, width, color= colors[2], yerr=_10_cis)
                _12_rects = ax.bar(ind + 3 * width, _12_means, width, color= colors[3], yerr=_12_cis)
                _30_rects = ax.bar(ind + 4 * width, _30_means, width, color= colors[4], yerr=_30_cis)
                _32_rects = ax.bar(ind + 5 * width, _32_means, width, color= colors[5], yerr=_32_cis)
                ax.set_ylabel('Ave. Loss per user guess')
            elif plt_type == "acc":
                plt.ylim(0.0,1.0)
                _00_rects = ax.bar(ind, _00_acc, width, color= colors[0])
                _02_rects = ax.bar(ind + width, _02_acc, width, color= colors[1])
                _10_rects = ax.bar(ind + 2 * width, _10_acc, width, color= colors[2])
                _12_rects = ax.bar(ind + 3 * width, _12_acc, width, color= colors[3])
                _30_rects = ax.bar(ind + 4 * width, _30_acc, width, color= colors[4])
                _32_rects = ax.bar(ind + 5 * width, _32_acc, width, color= colors[5])
                ax.set_ylabel('model accuracy')
            else:
                raise BaseException("tmp")
            ax.set_xticks(ind + (1.5 * width))
            ax.set_xticklabels(tuple(conditions))
            ax.legend((_00_rects[0], _10_rects[0], _30_rects[0]), ('sm', 'vm', 'cm'))
            ax.set_title('Model Comparison(' + plt_type + ')')
            #autolabel(_00_rects)
            #autolabel(_10_rects)
            #autolabel(_30_rects)
            #autolabel(_32_rects)
            plt.savefig('./images/model-compare-' + _set_name + '.' + plt_type + '.pdf', dpi=200, bbox_inches='tight', pad_inches=0)

