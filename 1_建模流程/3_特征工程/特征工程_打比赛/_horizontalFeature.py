# -*- coding: utf-8 -*-
"""
__title__ = '_horizongtalFeature'
__author__ = 'BinZhou'
__mtime__ = '2018/7/26'
"""
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import LabelEncoder

class HorizongtalFeature(object):
    def __init__(self):
        pass

    # 1. 对数值型特征，先进行计数和排序（横向衍生两列特征）
    @staticmethod
    def get_feats_vcrank(df, feat='', return_labelencoder=False):
        # 0.1 计数特征 value_counts
        ftr_ = df[feat].value_counts()
        ftr_ = pd.DataFrame(list(zip(ftr_.index,ftr_.values)),columns=[feat,feat+'_'+'vcounts'])
        df = df.merge(ftr_, 'left', on=feat)
        # 0.2 排序特征
        le = LabelEncoder()
        ftr_ = le.fit_transform(df[feat])
        df[feat+'_'+'rank'] = ftr_
        if return_labelencoder:
            return le, df
        return df

    # 2. 针对同类特征群（比如消费，浏览记录等）横向扩展，计算一些统计量作为特征
    @staticmethod
    def get_feats_syndrome(df, feat_cols=None):
        df = df.copy()
        _df = df[feat_cols]

        funcs = ['count', 'min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem', 'skew']
        for f in funcs:
            df['row_' + f] = _df.__getattr__(f)(1)
        if len(feat_cols) > 3:
            df['row_kurt'] = _df.kurt(1)
        df['row_q1'] = _df.quantile(0.25, 1)
        df['row_q3'] = _df.quantile(0.75, 1)
        df['row_q3_q1'] = df['row_q3'] - df['row_q1']
        df['row_max_min'] = df['row_max'] - df['row_min']
        df['row_cv'] = df['row_std'] / (df['row_mean'] + 10 ** -8)  # 变异系数
        df['row_cv_reciprocal'] = df['row_mean'] / (df['row_std'] + 10 ** -8)
        return df

    # 3. 多项式特征Polynomial

    # 4. 时间特征（未聚合）
    @staticmethod
    def get_feats_time(data, group=None, feats=None, ts='ts'):
        """
        (1) 输入的data需要先根据group和ts进行排序, data.sort_values([group,ts])
        (2) 时间形式2018-07-25，必须为时间类型 pd.to_datetime('2018-07-25')
        时间的聚合特征同数值型
        与时间相关特征的特征衍生的非聚合特征
        :param data:
        :param group: "id"
        :param feats: numerical features name
        :param ts:
        :return
        """
        print('time continuous ...')
        data[ts + '_year'] = data[ts].apply(lambda x: x.year)
        data[ts + '_month'] = data[ts].apply(lambda x: x.month)
        data[ts + '_day'] = data[ts].apply(lambda x: x.day)
        data[ts + '_weekday'] = data[ts].apply(lambda x: x.weekday())
        data[ts + '_diff'] = data.groupby(group)[ts].diff().apply(lambda x: x.days).fillna(0)  ##########
        # transform是对一列进行操作
        data[ts + '_time_interval'] = data[ts].transform(lambda x: x.max() - x).apply(lambda x: x.days)
        if feats:  # 对时间特征可用数值特征平均编码
            print("ts_average_encoding ...")
            gr = data.groupby(ts)
            for i in tqdm_notebook(feats):
                data['ts_average_encoding_' + i] = gr[i].transform('mean')  # median

            print("feats diff ...")
            gr = data.groupby(group)
            for i in tqdm_notebook(feats):  # 数值特征也可以按时间顺序进行差分
                data['diff_' + i] = gr[i].diff().fillna(0)
        return data
