# -*- coding: utf-8 -*-
"""
__title__ = 'Pickle'
__author__ = 'JieYuan'
__mtime__ = '2018/7/24'
"""

import pickle

from ..utils.decorator import execution_time


class Pickle(object):
    """
    https://blog.csdn.net/justin18chan/article/details/78516452
    json，用于字符串 和 python数据类型间进行转换:
        json只能处理简单的数据类型，比如字典，列表等，不支持复杂数据类型，如类等数据类型。
    """

    @execution_time
    @staticmethod
    def save_with_pickle(obj, file):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    @execution_time
    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    @execution_time
    @staticmethod
    def save_with_hdf(df, file):
        """方便下次快速读取"""
        df.to_hdf(file, 'w', complib='blosc', complevel=8)
