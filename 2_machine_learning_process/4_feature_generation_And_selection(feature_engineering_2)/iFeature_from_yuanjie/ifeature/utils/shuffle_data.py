# -*- coding: utf-8 -*-
"""
__title__ = 'shuffle_data'
__author__ = 'JieYuan'
__mtime__ = '2018/7/24'
"""
import numpy as np

from sklearn.utils import shuffle

def shuffle_data(X, y):
    s = np.arange(len(X))
    np.random.shuffle(s)
    X = X[s]
    y = y[s]
    return X, y



print(shuffle(np.arange(10), np.arange(10)))