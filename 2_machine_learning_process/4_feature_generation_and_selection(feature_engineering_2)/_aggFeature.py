# -*- coding: utf-8 -*-
"""
__title__ = '_aggFeature'
__author__ = 'BinZhou'
__mtime__ = '2018/7/27'
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from itertools import combinations


class AggFeature(object):

    def __init__(self):
        pass

    # 1. 可以未聚合之前或聚合之后使用，都可以
    @staticmethod
    def get_feats_vectors(X, vectorizer='TfidfVectorizer', tokenizer=None, ngram_range=(1, 1), max_features=None):
        """
        :param X: pd.Series
        :param vectorizer: 'TfidfVectorizer' or 'CountVectorizer'
        :param tokenizer: lambda x: x.split(',')
        :param ngram_range:
        :param max_features:
        :return:
        """
        vectorizer = text.__getattribute__(vectorizer)
        vectorizer = vectorizer(lowercase=False, tokenizer=tokenizer, ngram_range=ngram_range,
                                max_features=max_features)
        vectorizer.fit(X)
        return vectorizer

    # 2. 类别型特征（聚合）
    @staticmethod
    def get_feats_desc_cat(data, group='ID', feats=['feat1', ]):
        df = pd.DataFrame()
        for col_name in tqdm(feats):
            gr = data.groupby(group)[col_name]
            def _func():
                get_max_min = lambda x : np.max(x) - np.min(x)
                get_category_density = lambda x : pd.Series.nunique(x)*1.0 / pd.Series.count(x)
                # get_mode = lambda x : max(pd.Series.mode(x)) # 可能返回多个mode，取最大的那个mode
                get_mode = lambda x : x.value_counts().index[0]

                df = gr.agg({col_name + '_' + 'count':'count',
                             col_name + '_' + 'nunique':'nunique',
                             col_name + '_' + 'max':'max',
                             col_name + '_' + 'min':'min',
                             col_name + '_' + 'max_min':get_max_min,
                             col_name + '_' + 'mode':get_mode,
                             col_name + '_' + 'category_density':get_category_density}).reset_index()
                return df

            if col_name == feats[0]:
                col_name = group + '-' + col_name
                df = _func()
            else:
                col_name = group + '-' + col_name
                df = df.merge(_func(), 'left', group).fillna(0)
        return df

    # 3. 数值型特征（聚合）
    @staticmethod
    def get_feats_desc_numeric(data, group='ID', feats=['feat1', ]):
        """
        data未聚合
        时间特征差分后当数值型特征
        """
        print("There are %s features..."%str(len(feats)))
        df = pd.DataFrame()
        for col_name in tqdm(feats, desc='get_feature_desc'):
            gr = data.groupby(group)[col_name]

            def _func():
                q1_func = lambda x : x.quantile(0.25)
                q3_func = lambda x : x.quantile(0.75)
                get_max_min = lambda x : np.max(x) - np.min(x)
                get_q3_q1 = lambda x : x.quantile(0.75) - x.quantile(0.25)
                get_cov = lambda x : np.var(x)*1.0 / (np.mean(x)+10**-8)
                get_cov_reciprocal = lambda x : np.mean(x)*1.0 / (np.var(x)+10**-8)
                # (new_feature_name, operation)
                df = gr.agg({col_name + '_' + 'count': 'count',
                             col_name + '_' + 'mean': 'mean',
                             col_name + '_' + 'std': 'std',
                             col_name + '_' + 'var': 'var',
                             col_name + '_' + 'min': 'min',
                             col_name + '_' + 'max': 'max',
                             col_name + '_' + 'median': 'median',
                             col_name + '_' + 'q1': q1_func,
                             col_name + '_' + 'q3': q3_func,
                             col_name + '_' + 'max_min': get_max_min,
                             col_name + '_' + 'q3_q1': get_q3_q1,
                             col_name + '_' + 'kurt': pd.Series.kurt,
                             col_name + '_' + 'skew': pd.Series.skew,
                             col_name + '_' + 'sem': pd.Series.sem,
                             col_name + '_' + 'sum': np.sum,
                             col_name + '_' + 'COV': get_cov,
                             col_name + '_' + 'COV_reciprocal': get_cov_reciprocal}).reset_index()
                return df
            if col_name == feats[0]:
                df = _func()
            else:
                df = df.merge(_func(), 'left', group).fillna(0)
        return df

    # 4. Grougby类别型特征（比如时间，性别等）计算其他数值型特征的均值，方差等等（交叉特征，特征表征）
    # 然后再根据某个字段聚合比如user
    @staticmethod
    def create_fts_from_catgroup_aggnumeric(data, feats=None, by='ts', second_group='user'):
        '''
        :param data: 需要聚合的df
        :param feats: 数值型特征s
        :param by: 第一次聚合并transform的类别型特征
        :param second_group: 第二次聚合的列比如user
        :return:
        '''
        data = data.copy()
        q1_func = lambda x: x.quantile(0.25)
        q3_func = lambda x: x.quantile(0.75)
        get_max_min = lambda x: np.max(x) - np.min(x)
        get_q3_q1 = lambda x: x.quantile(0.75) - x.quantile(0.25)
        get_cov = lambda x: np.var(x) * 1.0 / (np.mean(x) + 10 ** -8)
        get_cov_reciprocal = lambda x: np.mean(x) * 1.0 / (np.var(x) + 10 ** -8)
        func_list = [('count', 'count'),
                     ('mean', 'mean'),
                     ('std', 'std'),
                     ('var', 'var'),
                     ('min', 'min'),
                     ('max', 'max'),
                     ('median', 'median'),
                     ('q1_func', q1_func),
                     ('q3_func', q3_func),
                     ('q3_q1', get_q3_q1),
                     ('max_min', get_max_min),
                     ('get_cov', get_cov),
                     ('get_cov_reciprocal', get_cov_reciprocal)]
        new_feats = []
        if feats is not None:  # 对时间特征可用数值特征平均编码
            print("%s_encoding ..." % by)
            gr = data.groupby(by)
            for ft in tqdm(feats):
                for func_name, func in func_list:
                    new_feat = '{}_{}_encoding_'.format(by, func_name) + ft
                    data[new_feat] = gr[ft].transform(func)
                    new_feats.append(new_feat)
        data = AggFeature.get_feats_desc_numeric(data, group=second_group, feats=new_feats)
        return data

