# -*- coding: utf-8 -*-
"""
__title__ = 'CategoryFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/7/20'
"""
from sklearn.feature_extraction import text

from ..utils.decorator import execution_time


class CategoryFeature(object):
    def __init__(self):
        self.feat_cols = None

    @execution_time
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

    @execution_time
    def get_feats_agg_desc(self, df, group_col=None, agg_cols=None, verbose=1):
        """
        :param df:
        :param group_col:
        :param agg_cols: list
        :return:
        """
        print("There are %s agg feats..." % len(agg_cols))

        if verbose:
            idx = iter(range(len(agg_cols)))

            def mode(x):
                global max
                if max != x.name:
                    print('Compute %4s: %-s' % (idx.__next__(), x.name))
                    max = x.name
                return x.value_counts().index[0]
        else:
            def mode(x):
                return x.value_counts().index[0]

        funcs = ['count', 'nunique', 'max', 'min', mode]

        df = df.groupby(group_col)[agg_cols].agg(funcs)
        df.columns = ['_cat_'.join(i) for i in df.columns]

        for col in agg_cols:
            df[col + '_max_min'] = df[col + '_cat_max'] - df[col + '_cat_min']
            df[col + '_category_density'] = df[col + '_cat_nunique'] / df[col + '_cat_count']

        self.feat_cols = df.columns.tolist()
        return df.reset_index()
