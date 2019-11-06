# -*- coding: utf-8 -*-
"""
__title__ = 'NumericalFeature'
__author__ = 'JieYuan'
__mtime__ = '2018/7/26'
"""

"""
__doc__ == 'Automatically created module for IPython interactive environment'
"""
from ..utils.decorator import execution_time


class NumericalFeature(object):
    def __init__(self, df):
        self.df = df
        self.feat_cols = None

    @execution_time
    def get_feats_agg_desc(self, group_cols, agg_cols=None, verbose=0):
        """
        :param group_cols: list
        :param agg_cols: list: 默认除group_cols以外的列
        :return:
        """
        assert isinstance(group_cols, list)

        if agg_cols is None:
            agg_cols = [col for col in self.df.columns if col not in group_cols]
        else:
            assert isinstance(agg_cols, list)

        df = self.df[group_cols + agg_cols]

        print("There are %s agg feats...\n" % len(agg_cols))

        if verbose:
            idx = iter(range(len(agg_cols)))

            def q1(x):
                global max
                if max != x.name:
                    print('Compute %4s: %-s' % (idx.__next__(), x.name))
                    max = x.name
                return x.quantile(0.25)
        else:
            def q1(x):
                return x.quantile(0.25)

        def q3(x):
            return x.quantile(0.75)

        def kurt(x):
            return x.kurt()

        def cv(x):
            return x.std() / (x.mean() + 10 ** -8)  # 变异系数

        funcs = ['count', 'min', 'mean', 'median', 'max', 'sum', 'std', 'var', 'sem', 'skew', kurt, q1, q3]

        df = df.groupby(group_cols)[agg_cols].agg(funcs)
        df.columns = ['_'.join(i) for i in df.columns]

        for col in agg_cols:
            df[col + '_max_min'] = df[col + '_max'] - df[col + '_min']
            df[col + '_q3_q1'] = df[col + '_q3'] - df[col + '_q1']
            df[col + '_cv'] = df[col + '_std'] / (df[col + '_mean'] + 10 ** -8)
            df[col + '_cv_reciprocal'] = 1 / (df[col + '_cv'] + 10 ** -8)

        self.feat_cols = list(df.columns)
        return df.reset_index()
