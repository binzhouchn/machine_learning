## 4. 特征选择

当数据预处理完成后，我们需要选择有意义的特征输入机器学习的算法和模型进行训练。通常来说，从两个方面考虑来选择特征：

- 特征是否发散：如果一个特征不发散，例如方差接近于0，也就是说样本在这个特征上基本上没有差异，这个特征对于样本的区分并没有什么用。

- 特征与目标的相关性：这点比较显见，与目标相关性高的特征，应当优先选择。除方差法外，本文介绍的其他方法均从相关性考虑。


根据特征选择的形式又可以将特征选择方法分为3种：


- Filter：过滤法，按照发散性或者相关性对各个特征进行评分，设定阈值或者待选择阈值的个数，选择特征。

- Wrapper：包装法，根据目标函数（通常是预测效果评分），每次选择若干特征，或者排除若干特征。

- Embedded：嵌入法，先使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。
类似于Filter方法，但是是通过训练来确定特征的优劣。　　

我们使用sklearn中的feature_selection库来进行特征选择。

---

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

