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



