### 3. 特征变换

**3.1 特征变换**

主要针对一些长尾分布的特征，需要进行幂变换或者对数变换，使得模型（LR或者DNN）能更好的优化。<br>
需要注意的是，Random Forest 和 GBDT 等模型对单调的函数变换不敏感。其原因在于树模型在求解分裂点的时候，只考虑排序分位点。

**3.2 特征编码**<br>

一、Binarization 特征二值化是将数值型特征变成布尔型特征
```python
from sklearn.preprocessing import Binarizer
bi = Binarizer(threshold=2)           # 设置阈值默认2.0  大于阈值设置为1 , 小于阈值设置为0
bi.fit_transform(df[['dd']])    # shape (1行,X列)
```

二、连续性变量划分份数，对定量特征多值化（分箱）
```python
import pandas as pd
pd.cut(df['app'],bins=5) # bin：int 在x范围内的等宽单元的数量
```

三、one-hot Encoding / Encoding categorical features
```python
pandas.get_dummies(df, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=False)
# dummy_na=False # 是否把 missing value单独存放一列
```

四、LabelEncoder
```python
# 对于Random Forest和GBDT等模型，如果类别特征存在较多的取值，可以直接使用 LabelEncoder 后的结果作为特征
from sklearn.preprocessing import LabelEncoder
LabelEncoder().fit_transform(X)
```

**3.2.1 特征编码遇到的问题**

对于取值较多（如几十万）的类别特征（ID特征）<br>
(1) 统计每个取值在样本中出现的频率，取 Top N 的取值进行 One-hot 编码，剩下的类别分到“其他“类目下，其中 N 需要根据模型效果进行调优<br>
(2) 统计每个 ID 特征的一些统计量（譬如历史平均点击率，历史平均浏览率）等代替该 ID 取值作为特征<br>
(3) 参考 word2vec 的方式，将每个类别特征的取值映射到一个连续的向量，对这个向量进行初始化，跟模型一起训练。训练结束后，可以同时得到每个ID的Embedding


