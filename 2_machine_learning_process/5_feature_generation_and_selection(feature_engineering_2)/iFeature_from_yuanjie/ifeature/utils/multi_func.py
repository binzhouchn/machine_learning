# -*- coding: utf-8 -*-
"""
__title__ = 'multi_func'
__author__ = 'JieYuan'
__mtime__ = '2018/8/29'
"""

from concurrent.futures import ProcessPoolExecutor

from tqdm import tqdm


def multi_func(func, iteration, n_jobs=16):
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        lst = pool.map(func, tqdm(iteration, 'Calculating ...'))
    return lst
