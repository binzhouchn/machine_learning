# -*- coding: utf-8 -*-
"""
__title__ = 'get_std_online'
__author__ = 'JieYuan'
__mtime__ = '2018/8/16'
"""
import numpy as np


def get_std_online(n, pairs=None):
    """参数对应
    pairs=[(r1, s1), (r2, s2)]
    """
    (r1, s1), (r2, s2) = pairs
    print((r1, s1), (r2, s2))
    _mean = ((s1 ** 2 - s2 ** 2) - (r1 ** 2 - r2 ** 2)) / (r2 - r1) / 2
    _square_mean = s1 ** 2 - r1 ** 2 + 2 * r1 * _mean

    _std = np.sqrt(_square_mean - _mean ** 2)
    return _mean, _std