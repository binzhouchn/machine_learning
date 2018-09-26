# -*- coding: utf-8 -*-
"""
__title__ = 'binning'
__author__ = 'JieYuan'
__mtime__ = '2018/8/28'
"""
import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from tqdm import tqdm


class Binning(object):
    def __init__(self, df: pd.DataFrame, label: str):
        assert label in df.columns
        self.label = label
        self.feats = [col for col in df.columns if col != label]
        self.X = df[self.feats]
        self.y = df[label].values
        self.node = {}

    def binning(self, return_X_y=True, n_jobs=16):
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            lst = pool.map(self.__binning, tqdm(self.feats, 'Binning ...'), chunksize=1)
        _data = np.column_stack(lst)
        if return_X_y:
            _data = pd.DataFrame(np.column_stack((_data, self.y)), columns=self.feats + [self.label])
        return _data

    def __binning(self, feat):
        """
        类别型合并分箱待补充 ...
        :param feat:
        :return:
        """
        _X = self.X[[feat]].values
        clf = LGBMClassifier(num_leaves=20, n_estimators=1, learning_rate=0.1)
        model = clf.fit(_X, self.y)
        rst = model.predict(_X, pred_leaf=True)
        js = json.dumps(model.booster_.dump_model()['tree_info'][0]['tree_structure'])
        self.node[feat] = sorted(map(float, re.findall(r'"threshold": (.*?),', js)))
        return rst
