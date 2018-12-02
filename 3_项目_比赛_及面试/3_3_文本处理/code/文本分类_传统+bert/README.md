# 文本分类（传统+BERT）

```python
import numpy as np
import pandas as pd
import re
import copy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import gensim
from gensim.models.word2vec import Word2Vec
from m import BOW
```

## tfidf+lr

```python
# data['comment_text'] 数据格式
# 'explanation why the edits made under my username'
# 'more i can't make any real suggestions'

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import KFold

class Tokenizer:
    def __init__(self):
        pass
    def __call__(self, line):
        tokens = line.split()
        return tokens

vectorizer = TfidfVectorizer(tokenizer=Tokenizer())
tfidf = vectorizer.fit_transform(data['comment_text'])
# 训练集和验证集划分 4:1
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(data.label):
    pass
train_X, train_y = tfidf[train_idx], np.array(data.label)[train_idx]
val_X, val_y = tfidf[val_idx], np.array(data.label)[val_idx]

lr = LogisticRegression(solver='lbfgs',max_iter=1000)
lr.fit(train_X, train_y)
# 预测验证集auc
val_pred = lr.predict_proba(val_X)[:,1]
print('auc: ', roc_auc_score(val_y, val_pred))
val_pred = lr.predict(val_X)
print('acc: ', accuracy_score(val_y, val_pred))
#auc:  0.9713322600236631
#acc:  0.9566647866140252
```

## d