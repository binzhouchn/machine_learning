# -*- coding: utf-8 -*-
"""
__title__ = '_horizongtalFeature'
__author__ = 'BinZhou'
__mtime__ = '2018/7/27'
"""
import pandas as pd
from tqdm import tqdm, tqdm_notebook
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from itertools import combinations

# 未聚合前的横向特征扩展

class HorizongtalFeature(object):

    def __init__(self):
        pass

    # 1. 对类别型特征，先进行计数和排序（每个特征横向衍生两列特征）value_counts和LabelEncoder
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
        :param data: np.array or pd.DataFrame, dataframe should be reindexed from 0 TO n
        :param feats: columns names
        :param degree:
        :return: df
        """
        df = data.copy()
        poly = PolynomialFeatures(degree, include_bias=False)
        df = poly.fit_transform(df[feats])

        if return_df:
            df = pd.DataFrame(df, columns=poly.get_feature_names(feats))
            df.drop(feats,axis=1,inplace=True)
            data = pd.concat([data,df],axis=1)
        if return_poly:
            return poly, data
        return data

    # 4. 时间特征 with_group
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

    # 4. 时间特征 without_group
    def create_fts_from_timegroup(data, feats=None, ts='ts'):
        data = data.copy()
        q1_func = lambda x: x.quantile(0.25)
        q3_func = lambda x: x.quantile(0.75)
        get_max_min = lambda x: np.max(x) - np.min(x)
        get_q3_q1 = lambda x: x.quantile(0.75) - x.quantile(0.25)
        get_cov = lambda x: np.var(x) * 1.0 / (np.mean(x) + 10 ** -8)
        get_cov_reciprocal = lambda x: np.mean(x) * 1.0 / (np.var(x) + 10 ** -8)
        func_list = [('count', 'count'), ('mean', 'mean'), ('std', 'std'), ('var', 'var'), ('min', 'min'),
                     ('max', 'max'), ('median', 'median'), \
                     ('q1_func', q1_func), ('q3_func', q3_func), ('q3_q1', get_q3_q1), ('max_min', get_max_min),
                     ('get_cov', get_cov), ('get_cov_reciprocal', get_cov_reciprocal)]
        if feats is not None:  # 对时间特征可用数值特征平均编码
            print("%s_encoding ..." % ts)
            gr = data.groupby(ts)
            for ft in tqdm_notebook(feats):
                for func_name, func in func_list:
                    data['{}_{}_encoding_'.format(ts, func_name) + ft] = gr[ft].transform(func)
        return data

    # 5. 组合特征
    '''
    造组合特征之前，一定要看下数据分布情况，然后确定哪些特征进行组合：
    技巧一：组合特征的分布最好要一致（不一定）；
    技巧二：几个特征的数值加总为1或者某个数，这个有业务意义，具体看场景；
    技巧三：特征随机组合，然后看组合后与类别的logloss(或KL散度)或相关性；
    技巧四：
    '''
    @staticmethod
    def get_numeric_feats_comb(df, operations=['add','sub','mul','div'], feature_for_polyAndcomb=None):
        df = df.copy()
        # 加减乘除
        add = lambda a, b : a + b
        sub = lambda a, b : a - b
        mul = lambda a, b : a * b
        div = lambda a, b: a / (b+10**-8)
        for oper in tqdm_notebook(operations):
            for f1, f2 in combinations(feature_for_polyAndcomb,2):
                col_name = f1+oper+f2
                df[col_name] = eval(oper)(df[f1],df[f2])
        return df