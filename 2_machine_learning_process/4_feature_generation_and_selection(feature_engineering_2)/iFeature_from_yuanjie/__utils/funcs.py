# -*- coding: utf-8 -*-
"""
__title__ = 'funcs'
__author__ = 'JieYuan'
__mtime__ = '2018/8/22'
"""
from collections import Counter, OrderedDict


def max_index(lst):
    """列表中最小和最大值的索引"""
    return max(range(len(lst)), key=lst.__getitem__)


def most_freq(x):
    """查找列表中频率最高的值"""
    return max(set(x), key=x.count)  # key作用于set(x), 可类推出其他用法


"""检查两个字符串是不是由相同字母不同顺序组成"""
print(Counter('str1') == Counter('str2'))


def transpose_2d_array(lst):
    """转置二维数组"""
    return list(zip(*lst))


def unique_order(lst):
    """移除列表中的重复元素"""
    return list(OrderedDict.fromkeys(lst))
