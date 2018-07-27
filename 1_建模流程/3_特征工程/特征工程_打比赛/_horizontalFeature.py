# -*- coding: utf-8 -*-
"""
__title__ = '_horizongtalFeature'
__author__ = 'BinZhou'
__mtime__ = '2018/7/27'
"""
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures

# 未聚合前的横向特征扩展

class HorizongtalFeature(object):

    def __init__(self):
        pass

    # 1. 对数值型特征，先进行计数和排序（横向衍生两列特征）
    @staticmethod
    def get_feats_vcrank(df, feat='', return_labelencoder=False):
        # 只能一列列进来
        # 0.1 计数特征 value_counts
        ftr_ = df[feat].value_counts()
        ftr_ = pd.DataFrame(list(zip(ftr_.index,ftr_.values)),columns=[feat,feat+'_'+'vcounts'])
        df = df.merge(ftr_, 'left', on=feat)
        # 0.2 排序特征
        le = LabelEncoder()
        ftr_ = le.fit_transform(df[feat])
        df[feat+'_'+'rank'] = ftr_
        if return_labelencoder:
            return le, df # 返回的le可以对测试数据进行transform
        return df

    # 2. 针对同类特征群（比如消费，浏览记录等）横向扩展，计算一些统计量作为特征
    @staticmethod
    def get_feats_syndrome(df, syndrome_num=0, feat_cols=None): #有多个特征群的时候会用到syndrome_num编号
        df = df.copy()
        _df = df[feat_cols]

        buildin_funcs = ['count', 'min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem', 'skew']
        for f in buildin_funcs:
            df['horz'+str(syndrome_num)+'_'+f] = _df.__getattr__(f)(axis=1)
        if len(feat_cols) > 3: # 从公式来看峰度n要大于3
            df['horz'+str(syndrome_num)+'_'+'kurt'] = _df.kurt(axis=1)
        df['horz'+str(syndrome_num)+'_'+'q1'] = _df.quantile(0.25, axis=1)
        df['horz'+str(syndrome_num)+'_'+'q3'] = _df.quantile(0.75, axis=1)
        df['horz'+str(syndrome_num)+'_'+'q3_q1'] = df['horz'+str(syndrome_num)+'_'+'q3'] - df['horz'+str(syndrome_num)+'_'+'q1']
        df['horz'+str(syndrome_num)+'_'+'max_min'] = df['horz'+str(syndrome_num)+'_'+'max'] - df['horz'+str(syndrome_num)+'_'+'min']
        df['horz'+str(syndrome_num)+'_'+'COV'] = df['horz'+str(syndrome_num)+'_'+'std'] / (df['horz'+str(syndrome_num)+'_'+'mean'] + 10 ** -8)  # 变异系数C.O.V
        df['horz'+str(syndrome_num)+'_'+'COV_reciprocal'] = df['horz'+str(syndrome_num)+'_'+'mean'] / (df['horz'+str(syndrome_num)+'_'+'std'] + 10 ** -8)
        return df

    # 3. 多项式特征Polynomial
    @staticmethod
    def get_feats_poly(data, feats=None, degree=2, return_df=True, return_poly=False):
        """PolynomialFeatures
        :param data: np.array or pd.DataFrame
        :param feats: columns names
        :param degree:
        :return: df
        """
        poly = PolynomialFeatures(degree, include_bias=False)
        data = poly.fit_transform(data[feats])

        if return_df:
            data = pd.DataFrame(data, columns=poly.get_feature_names(feats))
        if return_poly:
            return poly, data
        return data

    # 4. 时间特征
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
