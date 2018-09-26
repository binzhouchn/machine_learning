# -*- coding: utf-8 -*-
"""
__title__ = 'get_class'
__author__ = 'JieYuan'
__mtime__ = '2018/8/21'
"""

import numpy as np

def get_class(y_pred, n_pos):
    """
    :param y_pred: clf.predict_proba(X_test)[:, 1]
    :param n_pos: 通过训练集估计测试集个数
    :return:
    """
    threshold = sorted(y_pred, reverse=True)[n_pos] # 若果需要召回大，阈值可放小点
    y_pred = np.where(y_pred > threshold, 1, 0)
    return y_pred