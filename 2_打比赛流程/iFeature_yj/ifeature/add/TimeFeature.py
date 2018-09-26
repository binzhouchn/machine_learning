# -*- coding: utf-8 -*-
"""
__title__ = '_TimeFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/7/22'
"""

import numpy as np
import pandas as pd
from ..utils.decorator import execution_time

class TimeFeature(object):
    """doc"""
    def __init__(self):
        self.feat_cols = None
        self.feat_cat_cols = None  # 时间离散特征

    @execution_time
    def get_feats_time(self, df, group_col=None, time_col=None, feat_cols=None):
        """
        :param df:
        :param group_col:
        :param time_col: ts.astype('datetime64[ns]')
        :param feat_cols: list
        :return:
        """
        ts_cols = [time_col + i for i in ['_year', '_month', '_day', '_weekday']]
        self.feat_cat_cols = ts_cols

        print('Time Explodes Into Year/Month/Day ...')

        def explode_time(x):
            return [x.year, x.month, x.day, x.weekday()]

        ts = np.row_stack(df[time_col].apply(explode_time))
        _df = pd.DataFrame(ts, columns=ts_cols)
        df = pd.concat((df, _df), 1)

        print('Time Diff ...')
        df[time_col + '_diff'] = df.groupby(group_col)[time_col].diff().apply(lambda x: x.days).fillna(0)
        ts_diff_cols = [time_col + '_diff']

        print('Time Interval ...')
        df[time_col + '_time_interval'] = df[time_col].transform(lambda x: x.max() - x).apply(lambda x: x.days)
        ts_interval_cols = [time_col + '_time_interval']

        if feat_cols:
            print("Feats Diff ...")
            _df = df.groupby(group_col)[feat_cols].diff().fillna(0) \
                .rename(columns={i: i + '_diff' for i in feat_cols})  # 提前按时间升序排列
            df = pd.concat((df, _df), 1)
            feat_diff_cols = _df.columns.tolist()

            print("Feats Average Encoding ...")
            _df = df.groupby(time_col)[feat_cols].transform('mean') \
                .rename(columns={i: i + '_avg_encoding' for i in feat_cols})  # median
            df = pd.concat((df, _df), 1)
            feat_avg_cols = _df.columns.tolist()
            self.feat_cols = ts_diff_cols + ts_interval_cols + feat_diff_cols + feat_avg_cols
        else:
            self.feat_cols = ts_diff_cols + ts_interval_cols
        return df
