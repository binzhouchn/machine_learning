# -*- coding: utf-8 -*-
"""
__title__ = '_get_agg_feats'
__author__ = 'from yj'
__mtime__ = '2018/7/26'
"""
import pandas as pd
from sklearn.feature_extraction import text
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import PolynomialFeatures

# 类别型特征（聚合）
class CategoryFeature(object):
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

    @staticmethod
    def get_feats_desc_cat(data, group='ID', feats=None):
        for col_name in tqdm_notebook(feats):
            gr = data.groupby(group)[col_name]
            def _func():
                get_max_min = lambda x : np.max(x) - np.min(x)
                get_category_density = lambda x : pd.Series.nunique(x)*1.0 / pd.Series.count(x)
                get_mode = lambda x : max(pd.Series.mode(x)) # 可能返回多个mode，取最大的那个mode
                
                df = gr.agg([(col_name + '_' + 'count','count'),(col_name + '_' + 'nunique','nunique'),(col_name + '_' + 'max','max'),\
                             (col_name + '_' + 'min','min'),(col_name + '_' + 'max_min',get_max_min),(col_name + '_' + 'mode',get_mode),\
                            (col_name + '_' + 'category_density','min')]).reset_index()
                return df

            if col_name == feats[0]:
                df = _func()
            else:
                df = df.merge(_func(), 'left', group).fillna(0)
        return df


# 数值型特征（聚合）
class NumericalFeature(object):

    def __init__(self):
        pass

    @staticmethod
    def get_feats_poly(data, feats=None, degree=2, return_df=True):
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
        return data

    @staticmethod
    def get_feats_desc(data, group='ID', feats=['feature1', ]):
        """
        data未聚合
        时间特征差分后当数值型特征
        
        """
        print("There are %s features..."%str(len(feats)))

        for col_name in tqdm_notebook(feats, desc='get_feature_desc'):

#             _columns = {i: col_name + '_' + i for i in ['count', 'mean', 'std', 'var', 'min', 'q1', 'median', 'q3', 'max']}
            gr = data.groupby(group)[col_name]

            def _func():
                q1_func = lambda x : np.quantile(x, q=0.25)
                q3_func = lambda x : np.quantile(x, q=0.75)
                get_max_min = lambda x : np.max(x) - np.min(x)
                get_q3_q1 = lambda x : np.quantile(x, q=0.75) - np.quantile(x, q=0.25)
                get_coef_of_var = lambda x : np.var(x)*1.0 / np.mean(x)
                # (new_feature_name, operation)
                df = gr.agg([(col_name+'_'+'count','count'), (col_name+'_'+'mean','mean'), (col_name+'_'+'std','std'),\
                             (col_name+'_'+'var','var'), (col_name+'_'+'min','min'), (col_name+'_'+'max','max'),\
                             (col_name+'_'+'median','median'), (col_name+'_'+'q1',q1_func), (col_name+'_'+'q3',q3_func), \
                             (col_name+'_'+'max_min',get_max_min), (col_name+'_'+'q3_q1',get_q3_q1), (col_name+'_'+'kurt',pd.Series.kurt), \
                             (col_name+'_'+'skew',pd.Series.skew), (col_name+'_'+'sem',pd.Series.sem), (col_name+'_'+'sum',np.sum), \
                             (col_name+'_'+'COV',get_coef_of_var)]).reset_index()         
                return df
            if col_name == feats[0]:
                df = _func()
            else:
                df = df.merge(_func(), 'left', group).fillna(0)
        return df













