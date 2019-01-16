# -*- coding: utf-8 -*-
"""
__title__ = 'data'
__author__ = 'JieYuan'
__mtime__ = '2018/8/29'
"""

import pandas as pd
from sklearn.datasets import load_iris, make_classification

class Data(object):
    def __init__(self):
        pass


    @property
    def iris_df(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names).assign(label=iris.target)
        return df
