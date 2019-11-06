# -*- coding: utf-8 -*-
"""
__title__ = 'parallel.py'
__author__ = 'JieYuan'
__mtime__ = '2018/8/10'
"""
"""
由于GIL限制，建议：IO密集的任务，用ThreadPoolExecutor；CPU密集任务，用ProcessPoolExcutor

https://www.cnblogs.com/kangoroo/p/7628092.html
https://stackoverflow.com/questions/19849551/compute-on-pandas-dataframe-concurrently
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd


class PandasParallel(object):
    """按行并行运算"""

    def __init__(self, df):
        self.df = df

    def apply(self, func, n_jobs=16):
        """
        仅适合按行运算 df.apply(func, axis=0) or df.median(1)
        :param func: 不支持lambda
            def func(x):
                _df = x[1]
                return _df.median(1)
        :return: 由Serise拼接，返回pandas.core.series.Series
        """
        groups = self.__grouping(n_jobs)
        gr = self.df.groupby(groups)

        with ProcessPoolExecutor(n_jobs) as pool:
            return pd.concat(pool.map(func, gr))

    def __grouping(self, n_jobs):
        nrows = len(self.df)
        return np.arange(nrows) // n_jobs
