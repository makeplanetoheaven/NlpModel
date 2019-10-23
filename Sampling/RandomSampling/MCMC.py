# -*- coding: utf-8 -*-
'''
p:输入的概率分布，离散情况采用元素为概率值的数组表示
N:认为迭代N次马尔可夫链收敛
Nlmax:马尔可夫链收敛后又取的服从p分布的样本数
isMH:是否采用MH算法，默认为True
'''

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
from array import array


def MetropolisHastings(p, N=10000, Nlmax=10000, isMH=True):
    A = np.array([p for y in range(len(p))], dtype=np.float64)
    X0 = np.random.randint(len(p))
    count = 0
    samplecount = 0
    L = array("d", [X0])
    l = array("d")

    while True:
        X = int(L[samplecount])
        cur = np.argmax(np.random.multinomial(1, A[X]))
        count += 1
        if isMH:
            a = (p[cur] * A[cur][X]) / (p[X] * A[X][cur])
            alpha = min(a, 1)
        else:
            alpha = p[cur] * A[cur][X]
        u = np.random.uniform(0, 1)
        if u < alpha:
            samplecount += 1
            L.append(cur)
            if count > N:
                l.append(cur)
        if len(l) >= Nlmax:
            break
        else:
            continue
    La = np.frombuffer(L)
    la = np.frombuffer(l)
    return La, la


def count(q, n):
    L = array("d")
    l1 = array("d")
    l2 = array("d")
    for e in q:
        L.append(e)
    for e in range(n):
        l1.append(L.count(e))
    for e in l1:
        l2.append(e / sum(l1))
    return l1, l2
