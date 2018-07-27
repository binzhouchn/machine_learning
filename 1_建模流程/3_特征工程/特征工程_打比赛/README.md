# 0. 特征初筛

先对原始的特征进行初步筛选，筛选方案：<br>
对每个特征衍生特征，衍生方案：<br>
 - 对数值型特征，先进行计数和排序（横向衍生两列特征） 1.1.1
 - 针对同类特征群（比如消费，浏览记录等）横向扩展，计算一些统计量作为特征 1.2.1
 - 然后再衍生\[纵向]聚合特征 1.2.1, 1.2.2
 - 再进行特征减法
 - 最后单特征及单特征衍生特征跑一下模型比如lgb，看下效果（这里是AUC值）

取效果好的前几个特征，然后造多项式特征<br>
然后跑纵向扩展加法<br>
特征减法<br>

# 1. 特征加法

## 1.1 横向扩展加法（未聚合）

（1.1.1）计数和排序（衍生两列特征）value_counts和LabelEncoder

（1,1,2）针对同类特征群（比如消费，浏览记录等）横向扩展，计算一些统计量作为特征

（1.1.3）多项式特征Polynomial

（1.1.4）时间特征，Groupby ID，然后根据时间特征算一些未聚合的值比如年、月、日、星期几、组内日期差等

## 1.2 纵向扩展加法

（1.2.1）类别特征

（1.2.2）数值型特征


---

## 1.1 时间特征（未聚合）

算未聚合前的一些特征

Groupby ID，然后根据时间特征算一些未聚合的值比如年、月、日、星期几、组内日期差等

```python
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

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
```

## 1.2 类别特征（聚合）

算一些聚合特征

```python
from sklearn.feature_extraction import text
from tqdm import tqdm, tqdm_notebook


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

```

## 1.3 数值型特征（聚合）

算一些聚合特征

```python
import numpy as np
import pandas as pd
from tqdm import tqdm, tqdm_notebook

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
```


# 2. 特征减法

[_FeatureSelector.py](_FeatureSelector.py)

# 3. 效果验证

```python
import lightgbm as lgb
y = new_df2['LABEL'].copy()
X = new_df2.drop(['PERSONID','LABEL'],axis=1).copy()
lgb_data = lgb.Dataset(X, y)
params = {
    'boosting': 'gbdt', # 'rf', 'dart', 'goss'
    'application': 'binary', # 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
    'learning_rate': 0.01,
    'max_depth': -1,
    'num_leaves': 2 ** 7 - 1,

    'min_split_gain': 0,
    'min_child_weight': 1,

    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'lambda_l1': 0,
    'lambda_l2': 1,

    'scale_pos_weight': 1,
    'metric': 'auc',
    'num_threads': 32,
}
lgb.cv(
    params,
    lgb_data,
    num_boost_round=2000,
    nfold=5,
    stratified=False, # 回归一定是False
    early_stopping_rounds=100,
    verbose_eval=50,
    show_stdv=True)
```



