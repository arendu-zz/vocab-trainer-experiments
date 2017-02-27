#!/usr/bin/env python
__author__ = 'arenduchintala'
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    sns.set_style('whitegrid')
    #2.81
    #{1: 21, 2: 27, 3: 26, 4: 10, 5: 8, 6: 8}
    width = 0.3
    y = [21, 27, 26, 10, 8, 8, 0]
    fig, ax = plt.subplots()
    quiz_score = np.array([1, 2, 3, 4, 5, 6, 7])
    ax.set_ylabel('Count of users')
    ax.set_xlabel('Num correct quiz')
    ax.set_xticks(quiz_score + (0.5 * width))
    ax.set_xticklabels(tuple(quiz_score))
    ax.set_xticks(quiz_score + (0.5 * width))
    rects = ax.bar(quiz_score, y, width)
    plt.title('User performance on Quiz')
    plt.savefig('./images/user-performance-hist.pdf', dpi=200, bbox_inches='tight', pad_inches=0)
