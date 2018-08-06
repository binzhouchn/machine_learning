# 1. 总体流程（分类）

## 特征衍生方案（总）

流程如下：<br>
**横向未聚合衍生**<br>
 - （1）基础特征
 - （2）针对同类特征群（比如消费，浏览记录等）横向扩展，计算一些统计量作为特征 2.1.2
 - （3）时间衍生特征 2.1.4
 - （4）多项式特征 2.1.3
 - （5）组合特征（观察数据并进行加减乘除）
 - （1）+（2）+（3）+（4）+（5）拼接整合
 - 然后对拼接整合后中有类别型特征，先进行计数和排序（针对每个类别型特征再横向衍生两列特征） 2.1.1
 - 相关性筛一下（特征减法）
**纵向聚合衍生**<br>
 - 再衍生\[纵向]聚合特征(类别特征和数值型特征) 2.2.1, 2.2.2
 - 分别针对类别和数值型相关性筛一下（特征减法），拼接后再筛选一下
 - 整体跑一下模型比如lgb，看下效果（这里是AUC值）

## 特征初筛

（i）可以看下每个RobustScaler后的特征变量与类别之间的相关性<br>
 - 越负相关，区分度越高
 - 越正相关，区分度也越高
 - 取强正或负相关的前几个特征，然后造多项式特征 2.1.3 和组合特征（加减乘除）

（ii）最后单特征及单特征衍生特征跑一下模型比如lgb，看下效果（这里是AUC值）<br>
 - 特征auc低的比如小于0.7(看情况)的可以去掉
 - 取效果好的前几个特征(高auc和高方差)，然后造多项式特征 2.1.3 和组合特征（加减乘除）

# 2. 特征加法

## 2.1 横向扩展加法（未聚合）

（2.1.1）计数和排序（衍生两列特征）value_counts和LabelEncoder

（2,1,2）针对同类特征群（比如消费，浏览记录等）横向扩展，计算一些统计量作为特征

（2.1.3）多项式特征Polynomial

（2.1.4）时间特征，Groupby ID，然后根据时间特征算一些未聚合的值比如年、月、日、星期几、组内日期差等

（2.1.5） tfidf或counter特征（未聚合，比如每个用户每天的对话不拼接）

[_horizontalFeature](_horizontalFeature.py)

## 2.2 纵向扩展加法（聚合）

（2.2.1）类别特征

（2.2.2）数值型特征

（2.2.3） tfidf或counter特征（聚合，比如每个用户对话拼接以后再算）

[_aggFeature](_aggFeature.py)

# 3. 特征减法

[_FeatureSelector.py](_FeatureSelector.py)

# 4. 效果验证

```python
import lightgbm as lgb

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

# 5. 数据存取和模型存取

```python
## 数据存储
import os
import pandas as pd

def df_save(df, hdf_path=None):
    assert hdf_path is not None
    print("save h5 ...")
    df.to_hdf(hdf_path, 'w', complib='blosc', complevel=6)

## 数据读入
def df_read(hdf_path):
    if os.path.isfile(hdf_path):
        print("read h5 ...")
        df = pd.read_hdf(hdf_path)
        return df
```
```python
## 模型存取
import pickle

class Pickle(object):
    """
    https://blog.csdn.net/justin18chan/article/details/78516452
    json，用于字符串 和 python数据类型间进行转换:
        json只能处理简单的数据类型，比如字典，列表等，不支持复杂数据类型，如类等数据类型。
    """

    @staticmethod
    def save(obj, file):
        with open(file, 'wb') as f:
            pickle.dump(obj, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)
```

# 2. 总体流程（回归）

数据勘察和预处理之后，保留有用的特征；
然后进行回归特征工程（特征衍生和修剪）:

## 1. 缺失值处理（如果有）

## 2. 把一些数值型（但实际是类别型）的变量转成类别型

比如房屋号，房间号，时间（年，月）等astype(str)

## 3. 探索数值型，结合业务，进行特征组合

## 4. Grougby类别型特征（比如时间，性别等）计算其他数值型特征的均值，方差等等

## 5. 类别型进行LabelEncoder，that may contain information in their ordering set

## 6. 类别型进行one-hot编码

## 7. 再看下数值型特征（包括衍生的）是否有偏（一般大于0.75为有偏）

如果为长尾分布（偏度比较大）并且考虑使用线性模型，可以对变量进行幂变换或者对数变换或者box-cox进行转换。<br>
以下是box-cox进行转换代码<br>
```python
# 先看下哪些特征偏度比较大，从大到小排序
data[num_fts].apply(lambda x: x.skew()).sort_values(ascending=False)
# 然后进行box-cox转换
from scipy.special import boxcox, boxcox1p
for feat in num_fts:
    #all_data[feat] += 1
    data[feat] = boxcox1p(data[feat], 0.15)
```

## 8. 特征减法

## 9. 建模


[回归kaggle top4参考](https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook)
