# -*- coding: utf-8 -*-
"""
__title__ = '_aggFeature'
__desc__ = '聚合衍生特征'
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


#refered from [https://www.kaggle.com/lucasmorin/feature-engineering-2-aggregation-functions] optiver波动率预测比赛
# Statistical Agregation，很全的各种聚合函数 - 针对金融序列
def _roll(a, shift):
    """ Roll 1D array elements. Improves the performance of numpy.roll()"""

    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])


def _get_length_sequences_where(x):
    """ This method calculates the length of all sub-sequences where the array x is either True or 1. """
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]


def _aggregate_on_chunks(x, f_agg, chunk_len):
    """Takes the time series x and constructs a lower sampled version of it by applying the aggregation function f_agg on
    consecutive chunks of length chunk_len"""

    return [
        getattr(x[i * chunk_len: (i + 1) * chunk_len], f_agg)()
        for i in range(int(np.ceil(len(x) / chunk_len)))
    ]


def _into_subchunks(x, subchunk_length, every_n=1):
    """Split the time series x into subwindows of length "subchunk_length", starting every "every_n"."""
    len_x = len(x)

    assert subchunk_length > 1
    assert every_n > 0

    # how often can we shift a window of size subchunk_length over the input?
    num_shifts = (len_x - subchunk_length) // every_n + 1
    shift_starts = every_n * np.arange(num_shifts)
    indices = np.arange(subchunk_length)

    indexer = np.expand_dims(indices, axis=0) + np.expand_dims(shift_starts, axis=1)
    return np.asarray(x)[indexer]


def set_property(key, value):
    """
    This method returns a decorator that sets the property key of the function to value
    """

    def decorate_func(func):
        setattr(func, key, value)
        if func.__doc__ and key == "fctype":
            func.__doc__ = (
                    func.__doc__ + "\n\n    *This function is of type: " + value + "*\n"
            )
        return func

    return decorate_func


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series_log_return):
    return np.sqrt(np.sum(series_log_return ** 2))


def calc_wap(df):
    wap = (df['bid_price1'] * df['ask_size1'] + df['ask_price1'] * df['bid_size1']) / (
                df['bid_size1'] + df['ask_size1'])
    return wap


def median(x):
    return np.median(x)


def mean(x):
    return np.mean(x)


def length(x):
    return len(x)


def standard_deviation(x):
    return np.std(x)


def large_standard_deviation(x):
    if (np.max(x) - np.min(x)) == 0:
        return np.nan
    else:
        return np.std(x) / (np.max(x) - np.min(x))


def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan


def variance_std_ratio(x):
    y = np.var(x)
    if y != 0:
        return y / np.sqrt(y)
    else:
        return np.nan


def ratio_beyond_r_sigma(x, r):
    if x.size == 0:
        return np.nan
    else:
        return np.sum(np.abs(x - np.mean(x)) > r * np.asarray(np.std(x))) / x.size


def range_ratio(x):
    mean_median_difference = np.abs(np.mean(x) - np.median(x))
    max_min_difference = np.max(x) - np.min(x)
    if max_min_difference == 0:
        return np.nan
    else:
        return mean_median_difference / max_min_difference


def has_duplicate_max(x):
    return np.sum(x == np.max(x)) >= 2


def has_duplicate_min(x):
    return np.sum(x == np.min(x)) >= 2


def has_duplicate(x):
    return x.size != np.unique(x).size


def count_duplicate_max(x):
    return np.sum(x == np.max(x))


def count_duplicate_min(x):
    return np.sum(x == np.min(x))


def count_duplicate(x):
    return x.size - np.unique(x).size


def sum_values(x):
    if len(x) == 0:
        return 0
    return np.sum(x)


def log_return(list_stock_prices):
    return np.log(list_stock_prices).diff()


def realized_volatility(series):
    return np.sqrt(np.sum(series ** 2))


def realized_abs_skew(series):
    return np.power(np.abs(np.sum(series ** 3)), 1 / 3)


def realized_skew(series):
    return np.sign(np.sum(series ** 3)) * np.power(np.abs(np.sum(series ** 3)), 1 / 3)


def realized_vol_skew(series):
    return np.power(np.abs(np.sum(series ** 6)), 1 / 6)


def realized_quarticity(series):
    return np.power(np.sum(series ** 4), 1 / 4)


def count_unique(series):
    return len(np.unique(series))


def count(series):
    return series.size


# drawdons functions are mine
def maximum_drawdown(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) < 1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j - k


def maximum_drawup(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0

    series = - series
    k = series[np.argmax(np.maximum.accumulate(series) - series)]
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) < 1:
        return np.NaN
    else:
        j = np.max(series[:i])
    return j - k


def drawdown_duration(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0

    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j = k
    else:
        j = np.argmax(series[:i])
    return k - j


def drawup_duration(series):
    series = np.asarray(series)
    if len(series) < 2:
        return 0

    series = -series
    k = np.argmax(np.maximum.accumulate(series) - series)
    i = np.argmax(np.maximum.accumulate(series) - series)
    if len(series[:i]) == 0:
        j = k
    else:
        j = np.argmax(series[:i])
    return k - j


def max_over_min(series):
    if len(series) < 2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.max(series) / np.min(series)


def max_over_min_sq(series):
    if len(series) < 2:
        return 0
    if np.min(series) == 0:
        return np.nan
    return np.square(np.max(series) / np.min(series))


def mean_n_absolute_max(x, number_of_maxima=1):
    """ Calculates the arithmetic mean of the n absolute maximum values of the time series."""
    assert (
            number_of_maxima > 0
    ), f" number_of_maxima={number_of_maxima} which is not greater than 1"

    n_absolute_maximum_values = np.sort(np.absolute(x))[-number_of_maxima:]

    return np.mean(n_absolute_maximum_values) if len(x) > number_of_maxima else np.NaN


def count_above(x, t):
    if len(x) == 0:
        return np.nan
    else:
        return np.sum(x >= t) / len(x)


def count_below(x, t):
    if len(x) == 0:
        return np.nan
    else:
        return np.sum(x <= t) / len(x)


# number of valleys = number_peaks(-x, n)
def number_peaks(x, n):
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.
    """
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = x_reduced > _roll(x, i)[n:-n]

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= x_reduced > _roll(x, -i)[n:-n]
    return np.sum(res)


def mean_abs_change(x):
    return np.mean(np.abs(np.diff(x)))


def mean_change(x):
    x = np.asarray(x)
    return (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else np.NaN


def mean_second_derivative_central(x):
    x = np.asarray(x)
    return (x[-1] - x[-2] - x[1] + x[0]) / (2 * (len(x) - 2)) if len(x) > 2 else np.NaN


def median(x):
    return np.median(x)


def mean(x):
    return np.mean(x)


def length(x):
    return len(x)


def standard_deviation(x):
    return np.std(x)


def variation_coefficient(x):
    mean = np.mean(x)
    if mean != 0:
        return np.std(x) / mean
    else:
        return np.nan


def variance(x):
    return np.var(x)


def skewness(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)


def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


def root_mean_square(x):
    return np.sqrt(np.mean(np.square(x))) if len(x) > 0 else np.NaN


def absolute_sum_of_changes(x):
    return np.sum(np.abs(np.diff(x)))


def longest_strike_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x < np.mean(x))) if x.size > 0 else 0


def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x > np.mean(x))) if x.size > 0 else 0


def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size


def count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size


def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_maximum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN


# Test non-consecutive non-reoccuring values ?
def percentage_of_reoccurring_values_to_all_values(x):
    if len(x) == 0:
        return np.nan
    unique, counts = np.unique(x, return_counts=True)
    if counts.shape[0] == 0:
        return 0
    return np.sum(counts > 1) / float(counts.shape[0])


def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()
    if np.isnan(reoccuring_values):
        return 0

    return reoccuring_values / x.size


def sum_of_reoccurring_values(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)


def sum_of_reoccurring_data_points(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


def ratio_value_number_to_time_series_length(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size


def abs_energy(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.dot(x, x)


def quantile(x, q):
    if len(x) == 0:
        return np.NaN
    return np.quantile(x, q)


# crossing the mean ? other levels ?
def number_crossing_m(x, m):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    # From https://stackoverflow.com/questions/3843017/efficiently-detect-sign-changes-in-python
    positive = x > m
    return np.where(np.diff(positive))[0].size


def maximum(x):
    return np.max(x)


def absolute_maximum(x):
    return np.max(np.absolute(x)) if len(x) > 0 else np.NaN


def minimum(x):
    return np.min(x)


def value_count(x, value):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if np.isnan(value):
        return np.isnan(x).sum()
    else:
        return x[x == value].size


def range_count(x, min, max):
    return np.sum((x >= min) & (x < max))

# Lambda functions to facilitate application
count_above_0 = lambda x: count_above(x,0)
count_above_0.__name__ = 'count_above_0'

count_below_0 = lambda x: count_below(x,0)
count_below_0.__name__ = 'count_below_0'

value_count_0 = lambda x: value_count(x,0)
value_count_0.__name__ = 'value_count_0'

count_near_0 = lambda x: range_count(x,-0.00001,0.00001)
count_near_0.__name__ = 'count_near_0_0'

ratio_beyond_01_sigma = lambda x: ratio_beyond_r_sigma(x,0.1)
ratio_beyond_01_sigma.__name__ = 'ratio_beyond_01_sigma'

ratio_beyond_02_sigma = lambda x: ratio_beyond_r_sigma(x,0.2)
ratio_beyond_02_sigma.__name__ = 'ratio_beyond_02_sigma'

ratio_beyond_03_sigma = lambda x: ratio_beyond_r_sigma(x,0.3)
ratio_beyond_03_sigma.__name__ = 'ratio_beyond_03_sigma'

number_crossing_0 = lambda x: number_crossing_m(x,0)
number_crossing_0.__name__ = 'number_crossing_0'

quantile_01 = lambda x: quantile(x,0.1)
quantile_01.__name__ = 'quantile_01'

quantile_025 = lambda x: quantile(x,0.25)
quantile_025.__name__ = 'quantile_025'

quantile_075 = lambda x: quantile(x,0.75)
quantile_075.__name__ = 'quantile_075'

quantile_09 = lambda x: quantile(x,0.9)
quantile_09.__name__ = 'quantile_09'

number_peaks_2 = lambda x: number_peaks(x,2)
number_peaks_2.__name__ = 'number_peaks_2'

mean_n_absolute_max_2 = lambda x: mean_n_absolute_max(x,2)
mean_n_absolute_max_2.__name__ = 'mean_n_absolute_max_2'

number_peaks_5 = lambda x: number_peaks(x,5)
number_peaks_5.__name__ = 'number_peaks_5'

mean_n_absolute_max_5 = lambda x: mean_n_absolute_max(x,5)
mean_n_absolute_max_5.__name__ = 'mean_n_absolute_max_5'

number_peaks_10 = lambda x: number_peaks(x,10)
number_peaks_10.__name__ = 'number_peaks_10'

mean_n_absolute_max_10 = lambda x: mean_n_absolute_max(x,10)
mean_n_absolute_max_10.__name__ = 'mean_n_absolute_max_10'

#How to treat the first step ?
#to test immediately
get_first_ret = lambda x: np.log(x.iloc[0])
get_first_ret .__name__ = 'get_first_ret'

get_first_vol = lambda x: np.square(np.log(x.iloc[0]))
get_first_vol .__name__ = 'get_first_vol'

base_stats = [mean,sum,length,standard_deviation,variation_coefficient,variance,skewness,kurtosis]
higher_order_stats = [abs_energy,root_mean_square,sum_values,realized_volatility,realized_abs_skew,realized_skew,realized_vol_skew,realized_quarticity]
min_median_max = [minimum,median,maximum]
additional_quantiles = [quantile_01,quantile_025,quantile_075,quantile_09]
other_min_max = [absolute_maximum,max_over_min,max_over_min_sq]
min_max_positions = [last_location_of_maximum,first_location_of_maximum,last_location_of_minimum,first_location_of_minimum]
peaks = [number_peaks_2, mean_n_absolute_max_2, number_peaks_5, mean_n_absolute_max_5, number_peaks_10, mean_n_absolute_max_10]
counts = [count_unique,count,count_above_0,count_below_0,value_count_0,count_near_0]
reoccuring_values = [count_above_mean,count_below_mean,percentage_of_reoccurring_values_to_all_values,percentage_of_reoccurring_datapoints_to_all_datapoints,sum_of_reoccurring_values,sum_of_reoccurring_data_points,ratio_value_number_to_time_series_length]
count_duplicate = [count_duplicate,count_duplicate_min,count_duplicate_max]
variations = [mean_abs_change,mean_change,mean_second_derivative_central,absolute_sum_of_changes,number_crossing_0]
ranges = [variance_std_ratio,ratio_beyond_01_sigma,ratio_beyond_02_sigma,ratio_beyond_03_sigma,large_standard_deviation,range_ratio]
get_first = [get_first_ret, get_first_vol]

all_functions = base_stats + higher_order_stats + min_median_max + additional_quantiles + other_min_max + min_max_positions + peaks + counts + variations + ranges
