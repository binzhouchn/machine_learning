# -*- coding: utf-8 -*-
"""
__title__ = '_Encoder'
__author__ = 'JieYuan'
__mtime__ = '2018/7/26'
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

__all__ = [
    'LabelEncoder',  # 排序特征
    'CountEncoder',  # 计数特征
]

class CountEncoder(object):

    def __repr__(self):
        return 'ifeature.featadd.CountEncoder'

    def fit(self, df: pd.Series):
        self.counter = df.value_counts(sort=False).reset_index(name=df.name + '_counter')
        return self

    def fit_transform(self, df: pd.Series):
        self.counter = df.value_counts(sort=False).reset_index(name=df.name + '_counter')
        return df.to_frame('index').merge(self.counter, 'left').drop('index', 1).fillna(0)

    def transform(self, df: pd.Series):
        return df.to_frame('index').merge(self.counter, 'left').drop('index', 1).fillna(0)




