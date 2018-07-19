## 2. 数据预处理

通过特征提取，我们能得到未经处理的特征，这时的特征可能有以下问题：

- 不属于同一量纲：即特征的规格不一样，不能够放在一起比较。无量纲化可以解决这一问题。

- 信息冗余：对于某些定量特征，其包含的有效信息为区间划分，例如学习成绩，假若只关心“及格”或不“及格”，那么需要将定量的考分，转换成“1”和“0”表示及格和未及格。二值化可以解决这一问题。

- 定性特征不能直接使用：某些机器学习算法和模型只能接受定量特征的输入，那么需要将定性特征转换为定量特征。最简单的方式是为每一种定性值指定一个定量值，但是这种方式过于灵活，增加了调参的工作。通常使用哑编码的方式将定性特征转换为定量特征：假设有N种定性值，则将这一个特征扩展为N种特征，当原始特征值为第i种定性值时，第i个扩展特征赋值为1，其他扩展特征赋值为0。哑编码的方式相比直接指定的方式，不用增加调参的工作，对于线性模型来说，使用哑编码后的特征可达到非线性的效果。

- 存在缺失值：缺失值需要补充。

- 信息利用率低：不同的机器学习算法和模型对数据中信息的利用是不同的，之前提到在线性模型中，使用对定性特征哑编码可以达到非线性的效果。类似地，对定量变量多项式化，或者进行其他的转换，都能达到非线性的效果。　

我们使用sklearn中的preproccessing库来进行数据预处理，可以覆盖以上问题的解决方案。

---

### 一、标准化(Standardization)

公式：新数据 =（原数据- 均值）/ 标准差<br>
```python
from sklearn.preprocessing import StandardScaler
sds = StandardScaler()
sds.fit(X)

x_train_sds = sds.transform(X)
```

### 二、区间缩放法(MinMaxScaler) 特征缩放至特定范围 , default=(0, 1)

公式：新数据 =（原数据 - 最小值）/（最大值 - 最小值）<br>
```python
from sklearn.preprocessing import MinMaxScaler
mns = MinMaxScaler((0,1))
mns.fit(X)

x_train_mns = mns.transform(X)
```

### 三、归一化(正则化) Normalization

使单个样本具有单位范数的缩放操作。 经常在文本分类和聚类当中使用。<br>
```python
from sklearn.preprocessing import Normalizer
Normalizer().fit_transform([[1,2,3],[11,22,23]])
# 一般计算两个向量[1,2,3]，[11,22,23]的余弦相似的时候用的比较多
# 一般不对特征进行正则化处理
```

### 四、缺失值填充

2_数据预处理 -> 数据清洗中有介绍<br>
```python
from sklearn.preprocessing import Imputer
im =Imputer(strategy='mean') # mean, median, most_frequent
im.fit_transform(X)
```

