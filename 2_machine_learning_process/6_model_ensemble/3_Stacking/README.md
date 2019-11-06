## Stacking

 - **基于串行策略**：初级学习器与次级学习器之间存在依赖关系，初学习器的输出作为次级学习器的输入
 - **基本思路**
    - 先从初始训练集训练T个不同的初级学习器；
    - 利用每个初级学习器的输出构建一个次级数据集，该数据集依然使用初始数据集的标签；
    - 根据新的数据集训练次级学习器；
    - 多级学习器的构建过程类似

> 周志华-《机器学习》中没有将 Stacking 方法当作一种集成策略，而是作为一种结合策略，比如加权平均和投票都属于结合策略
 - 为了降低过拟合的风险，一般会利用**交叉验证**的方法使不同的初级学习器在**不完全相同的子集**上训练
 ```
    以 k-折交叉验证为例：
    - 初始训练集 D={(x_i, y_i)} 被划分成 D1, D2, .., Dk；
    - 记 h_t 表示第 t 个学习器，并在除 Dj 外的数据上训练；
    - 当 h_t 训练完毕后，有 z_it = h_t(x_i)；
    - T 个初级学习器在 x_i 上共产生 T 个输出；
    - 这 T 个输出共同构成第 i 个次级训练数据 z_i = (z_i1, z_i2, ..., z_iT)，标签依然为 y_i；
    - 在 T 个初级学习器都训练完毕后，得到次级训练集 D'={(z_i, y_i)}
```

## 通用
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC # not proba
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier # not proba
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from lightgbm.sklearn import LGBMClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

iris = load_iris()
X = iris.data
y = iris.target

# clf11 = RadiusNeighborsClassifier() # not proba
# clf12 = SVC() # not proba
clf1 = LogisticRegression()
clf2 = RandomForestClassifier()
clf3 = GradientBoostingClassifier()
clf4 = GaussianNB()
clf5 = KNeighborsClassifier()
clf6 = MLPClassifier()
clf7 = LGBMClassifier()
clf8 = XGBClassifier()

clfs = [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8]
lr = LogisticRegression()
```

## StackingClassifier
```python
sclf = StackingClassifier(classifiers=clfs, 
                          meta_classifier=lr, 
                          use_probas=True,
                          average_probas=False,
                          verbose=1)

scores = cross_val_score(sclf, X, y, cv=3, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
```

## StackingCVClassifier
```python


sclf = StackingCVClassifier(classifiers=clfs, 
                            meta_classifier=lr,
                            use_probas=True, 
                            cv=3, 
                            verbose=1)

skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # .split(X, y)
scores = cross_val_score(sclf, X, y, cv=skf, scoring='accuracy') 
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
```

## EnsembleVoteClassifier
```python
eclf = EnsembleVoteClassifier(clfs=clfs, voting='hard', weights=[1]*len(clfs))
scores = cross_val_score(eclf, X, y, cv=3, scoring='accuracy')
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
```
