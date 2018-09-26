# -*- coding: utf-8 -*-
"""
__title__ = 'RowFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/7/26'
"""
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from tqdm import tqdm

from ..utils.decorator import execution_time
from ..utils.parallel import PandasParallel


def func(x):
    """优化median"""
    return x[1].median(1)


class RowFeature(object):
    def __init__(self):
        self.feat_desc_cols = None
        self.feat_poly_cols = None

    @execution_time
    def get_feats_row_desc(self, df, feat_cols=None, prefix='row'):

        _df = df[feat_cols]

        if len(feat_cols) > 3:
            df[prefix + '_kurt'] = _df.kurt(1)

        df[prefix + '_median'] = PandasParallel(_df).apply(func, 32)

        funcs = ['count', 'min', 'mean', 'max', 'sum', 'std', 'var', 'sem', 'skew', None, None]
        for f in tqdm(funcs):
            if f:
                df[prefix + '_' + f] = _df.__getattr__(f)(1)

        df[prefix + '_q1'] = _df.quantile(0.25, 1)
        df[prefix + '_q3'] = _df.quantile(0.75, 1)
        df[prefix + '_q3_q1'] = df['row_q3'] - df['row_q1']
        df[prefix + '_max_min'] = df['row_max'] - df['row_min']
        df[prefix + '_cv'] = df['row_std'] / (df['row_mean'] + 1e-8)  # 变异系数
        df[prefix + '_cv_reciprocal'] = df['row_mean'] / (df['row_std'] + 1e-8)

        self.feat_desc_cols = [col for col in df.columns if col.startswith(prefix)]
        return df

    @execution_time
    def get_feats_row_poly(self, df, feats=None, degree=2, return_df=True):
        """PolynomialFeatures
        :param data: np.array or pd.DataFrame
        :param feats: columns names
        :param degree:
        :return: df
        """
        if feats is None:
            feats = df.columns

        poly = PolynomialFeatures(degree, include_bias=False)
        df = poly.fit_transform(df[feats])
        self.feat_poly_cols = poly.get_feature_names(feats)

        if return_df:
            df = pd.DataFrame(df, columns=self.feat_poly_cols)
        return df
