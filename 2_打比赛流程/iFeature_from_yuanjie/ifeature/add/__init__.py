# -*- coding: utf-8 -*-
"""
__title__ = '__init__.py'
__author__ = 'JieYuan'
__mtime__ = '2018/7/23'
"""

"""特征工程
特征最好不要超过1000
根据熵、距离、概率（比率）构造特征。
技巧：根据已有的各种统计量构造，比如卡方(x-E)/E、各种统计量 ...
    聚类特征
"""

from .Binning import Binning
from .CategoryFeature import CategoryFeature
from .NumericalFeature import NumericalFeature
from .RowFeature import RowFeature
from .TimeFeature import TimeFeature
