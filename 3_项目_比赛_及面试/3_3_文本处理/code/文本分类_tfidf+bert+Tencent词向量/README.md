# 文本分类（tfidf + bert + Tencent词向量）

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

## 1. tfidf+lr

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

## 2. bert句向量接xgb

[bert as service](https://github.com/hanxiao/bert-as-service)<br>
 
 - 先启动bert as service端口号是5555
 - 把service/client.py文件放到notebook启动目录下(也可以任何位置)
 - 导入BertClient
```python
from client import BertClient # 启动目录下
bc = BertClient()
# bc.encode(['我 是 中国 人'])
bc_X = data.comment_text.apply(lambda x : bc.encode([x])[0])
# train dev split
bc_X_train = bc_X[train_idx]
bc_y_train = data.label[train_idx]
bc_X_dev = bc_X[val_idx]
bc_y_dev = data.label[val_idx]

# 接xgb或lgb
from xgboost.sklearn import XGBClassifier
clf = XGBClassifier(
    n_estimators=17,
    learning_rate=0.05,
    max_depth=10,
    min_child_weight=1,
    gamma=0.3,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=12,
    scale_pos_weight=1,
    reg_lambda=1,
    seed=88)
model_xgb = clf.fit(np.array(bc_X_train.tolist()), bc_y_train.tolist())
# val on dev set
print('acc: ',accuracy_score(bc_y_dev.tolist(), model_xgb.predict(np.array(bc_X_dev.tolist()))))
print('auc: ', roc_auc_score(bc_y_dev.tolist(), model_xgb.predict_proba(np.array(bc_X_dev.tolist()))[:,1]))
```

## 3. 生成train和dev跑bert模型 run_classify.py

[google bert](https://github.com/google-research/bert/)<br>
[BERT在极小数据下带来显著提升的开源实现](https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247493161&idx=1&sn=58ddcd071602c42dda93275289311bb3&chksm=96ea39a9a19db0bf15df95cc9961064a3bab4e4a0b8d25a9bfb45c154942330b9cdb0abe0f4b&scene=0&xtrack=1#rd)<br>

```python
# comment不需要切分，用空格分开就行
tmp = data[['comment_text','label']]
tmp_train = tmp.loc[train_idx]
tmp_dev = tmp.loc[val_idx]
# 保存成tsv文件
tmp_train.to_csv('data/train.tsv',sep='\t',index=False)
tmp_dev.to_csv('data/dev.tsv',sep='\t',index=False)

# 这里就一列text输入，二分类问题
# 所以run_classify.py文件得改 MRPC任务(MrpcProcessor)
# 还有文件中各种路径
```
[run_classify.py](run_classifier_v1.py)

## 4. 用腾讯外部词+词向量

1. 把腾讯词向量存入mongodb中，需先安装mongodb<br>
2. 安装pymongo，连接启动的mongodb，创建数据库和集<br>
```python
import pymongo
client = pymongo.MongoClient(host='xx.xxx.x.xx', port=27017)
db = client.tencent_wv
my_set = db.test_set
# 删除 test_set
db.drop_collection('test_set')
```
```python
# 插入词向量
from tqdm import tqdm
def __reader():
    with open("/opt/common_files/Tencent_AILab_ChineseEmbedding.txt",encoding='utf-8',errors='ignore') as f:
        for idx, line in tqdm(enumerate(f), 'Loading ...'):
            ws = line.strip().split(' ')
            if idx:
                vec = [float(i) for i in ws[1:]]
                if len(vec) != 200:
                    continue
                yield {'word': ws[0], 'vector': vec}

rd = __reader()
# 插入
while rd:
    my_set.insert_one(next(rd))
# 查询
my_set.find_one({'word':'document'})
```
3. 把8百多万个单词提出来 作为外部词典<br>
```python
word_list = []
for x in tqdm(my_set.find()):
    word_list.append(x['word'])
# 写入文件
with open('../save/tencent_words.txt','w',encoding='utf-8') as f:
    for w in word_list:
        f.write(w + '\n')
```
4. 把文本词对应的词向量从mongodb中取出<br>
```python
# 读取word2idx
word2idx = np.load('../save/wd2idx.npy').tolist()
res = []
for x in tqdm(my_set.find()):
    if x['word'] in word2idx.keys():
        res.append(x)
d_ = {}
for x in res:
    d_[x['word']] = x['vector']
# embedding matrix
embedding_dim = 200
vocab_size = len(word2idx)
embedding_matrix = np.zeros((vocab_size+1,embedding_dim))
for key, value in tqdm(word2idx.items()):
    if key in d_.keys():
        embedding_matrix[value] = d_[key]
    else:
        embedding_matrix[value] = [0] * embedding_dim
```