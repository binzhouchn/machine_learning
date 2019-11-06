# 1. 特征衍生

【有agg的情况 聚合】（多列，id重复）

数值型特征：可以有统计特征(count sum avg)，排序特征和多项式特征<br>

类别型特征：count nunique category_density等

时间特征

[_aggFeature.py](_aggFeature.py)

---

【没有agg的情况 未聚合】（单列，id不重复）

 - 数值列 
    - 缺失值特征
    - 异常值特征：3sigma/箱型图，孤立森林(isolated forest)
    - 分箱
    - 多项式特征
 - 类别型特征 
    - 计数和排序特征
    - （交叉特征）：结合类别型特征对数值型进行编码
 - 时间特征
 - 组合特征（强特组合）
 
[_horizontalFeature.py](_horizontalFeature.py)

[可借鉴iFeature_from_yuanjie](iFeature_from_yuanjie)

# 2. 特征筛选

## 特征选择

当数据预处理完成后，我们需要选择有意义的特征输入机器学习的算法和模型进行训练。通常来说，从两个方面考虑来选择特征：

- 特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。

- 特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优先选择。除方差法外，本文介绍的其他方法均从相关性考虑。


根据特征选择的形式又可以将特征选择方法分为3种：


- Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。

- Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。

- Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。
类似于Filter方法，但是是通过训练来确定特征的优劣。　　

我们使用sklearn中的feature_selection库来进行特征选择。

### Embedded

推荐使用 feature importance, Tree-base > L1-base > ... //

(1) **基于惩罚项的特征选择法 Lasso**
```python
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(xdata,ydata)

# 利用模型进行筛选的方法
from sklearn.feature_selection import SelectFromModel
model = SelectFromModel(lasso,prefit=True)
x_new = model.transform(xdata)
```

(2) **基于树模型的特征选择法（常用）Tree-based feature selection**
```python
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(xdata,ydata)

rf.feature_importances_  # 非线性模型, 没有系数, 只有变量重要性!!!!
```

(3) **基于Univariate feature selection  单变量特征选择**
```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression
skb = SelectKBest(f_regression,k=10)
#skb = SelectPercentile(f_regression,percentile=10)
skb.fit_transform(xdata,ydata)
```

## 降维

- 当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大，训练时间长的问题，因此降低特征矩阵维度也是必不可少的。
- 常见的降维方法除了以上提到的基于L1惩罚项的模型以外，另外还有主成分分析法（PCA）和线性判别分析（LDA），线性判别分析本身也是一个分类模型。<br>
- PCA和LDA有很多的相似点，其本质是要将原始的样本映射到维度更低的样本空间中，但是PCA和LDA的映射目标不一样：`PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能。`所以说PCA是一种无监督的降维方法，而LDA是一种有监督的降维方法。

### 主成分分析(PCA)，计算协方差矩阵

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=15) # n_components : 设置留下来几列
newdata = pca.fit_transform(xdata)

#pca.explained_variance_            # 可解释方差
#pca.explained_variance_ratio_      # 可解释方差百分比
```

\*注意：PCA 前先将数据进行标准化，用z-score即可!!!

### 截断SVD(TruncatedSVD)

TruncatedSVD 原来N列 可以选择指定保留k列 , 降维<br>
```python
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=15, algorithm='randomized')
newdata = tsvd.fit_transform(xdata)
#n_components：int  , 输出数据的期望维度。
```

### 线性判别分析法(LDA) 有监督的降维

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit_transform(X_train, y_train)

lda.transform(X_test)
```

 