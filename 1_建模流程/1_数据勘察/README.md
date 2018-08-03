<h1 align = "left">:alien: 数据探索 :alien:</h1>

---
从两个方面进行分析

## 数据质量分析

 - 1. **描述性统计**<br>
数据类型、缺失值、异常值、重复数据、不一致的值等

## 数据特征分析

 - 1. **分析特征变量的分布**<br>
(1) 特征变量为连续值：如果为长尾分布并且考虑使用线性模型，可以对变量进行幂变换或者对数变换，或者常用的box-cox变换(skew>0.75)。<br>
(2) 特征变量为离散值：观察每个离散值的频率分布，对于频次较低的特征，可以考虑统一编码为“其他”类别。<br>
比如天气特征有晴、阴、雨、刮风、雪，如果刮风和雪的频次较低则可编码为'其他'

 - 2. **分析目标变量的分布**<br>
(1) 目标变量为连续值：查看其值域范围是否较大，如果较大，可以考虑对其进行对数变换，并以变换后的值作为新的目标变量进行建模（在这种情况下，需要对预测结果进行逆变换）。一般情况下，可以对连续变量进行Box-Cox变换。通过变换可以使得模型更好的优化，通常也会带来效果上的提升。<br>
(2)目标变量为离散值：如果类别分布不平衡，考虑是否需要上采样/下采样；如果目标变量在某个ID上面分布不平衡，在划分本地训练集和验证集的时候，需要考虑分层采样(Stratified Sampling)。<br>
下采样最好结合业务场景，(或业务规则)，让负样本急剧下降。

针对反欺诈场景，类别不平衡的情况：<br>
(i) 欠采样(用的比较多)<br>
(ii) 过采样(SMOTE: Synthetic Minority Over-sampling Technique)
```python
from imblearn.over_sampling import SMOTE
# SMOTE Technique (OverSampling) After splitting and Cross Validating
sm = SMOTE(ratio='minority', random_state=42)
Xsm_train, ysm_train = sm.fit_sample(original_Xtrain, original_ytrain)
```

 - 3. **分析特征变量和目标变量之间的关系**<br>
 (1) **对于回归问题**，<br>
 （i）类别型特征，画boxplot<br>
 ```python
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline 
fig = sns.boxplot(x='col_1', y="label", data=data)
fig.axis(ymin=0, ymax=100); # y轴设置
```
 （ii）数值型特征，画scatterplot<br>
```python
data.plot.scatter(x='col_1', y='label', ylim=(0, 100))
```
 (2) **对于分类问题**，<br>
 待补充

 - 4. **分析变量之间两两的分布和相关度**<br>
(1) 分析连续变量之间线性相关程度的强弱，可以用于发现高相关和共线性的特征。

 - 5. **周期性分析**<br>
 (1) 探索某个变量是否随着时间变化而呈现出某种周期变化趋势

[链接地址](https://m.sohu.com/a/139981834_116235)

---

```python
import DataFrameSummary # from pandas_summary import DataFrameSummary
dfs = DataFrameSummary(df)
```

- 列类型: dfs.columns_types <br>
```python
floating     9
integer     3
boolean        3
categorical 2
date        1
string        1
dtype: int64
```

- 列统计: dfs.columns_stats

```python
                      A            B        C          D          E 
counts             5802         5794     5781       5781       4617   
uniques            5802            3     5771        128        121   
missing               0            8       21         21       1185   
missing_perc         0%        0.14%    0.36%      0.36%     20.42%   
types            integer  categorical  floating    floating    floating 
```

- 相关系数矩阵: dfs.corr

|cor|a|b|c|d|
|:--:|:--:|:--:|:--:|:--:|
|a	|1.000000	|-0.109369  |0.871754	|0.817954|
|b	|-0.109369  |1.000000   |-0.420516  |-0.356544|
|c	|0.871754	|-0.420516  |1.000000	|0.962757|
|d	|0.817954	|-0.356544  |0.962757	|1.000000|


- 列汇总: 单列dfs['a'], 所有列dfs.summary()
```python
std                                                                 0.2827146
max                                                                  1.072792
min                                                                         0
variance                                                           0.07992753
mean                                                                0.5548516
5%                                                                  0.1603367
25%                                                                 0.3199776
50%                                                                 0.4968588
75%                                                                 0.8274732
95%                                                                  1.011255
iqr                                                                 0.5074956
kurtosis                                                            -1.208469
skewness                                                            0.2679559
sum                                                                  3207.597
mad                                                                 0.2459508
cv                                                                  0.5095319
zeros_num                                                                  11
zeros_perc                                                               0,1%
deviating_of_mean                                                          21
deviating_of_mean_perc                                                  0.36%
deviating_of_median                                                        21
deviating_of_median_perc                                                0.36%
counts                                                                   5781
uniques                                                                  5771
missing                                                                    21
missing_perc                                                            0.36%
types                                                                 numeric
Name: A, dtype: object
```
