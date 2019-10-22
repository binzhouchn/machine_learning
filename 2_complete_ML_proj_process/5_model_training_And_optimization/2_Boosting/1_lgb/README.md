<h1 align = "center">:helicopter: lgb常用参数 :running:</h1>

# 目录

[**1. lgb baseline模型**](#lgb_baseline模型)

[**2. lgb 自定义metric**](#lgb_自定义metric)

[**3. lgb常用参数**](#lgb常用参数)

[**4. oof**](#oof)

---

## lgb_baseline模型

[lgb_baseline模型](baseline_model.py)

baseline可以用杰哥封装的包<br>
```python
from iwork.models.classifier import BaselineLGB
from iwork.models import OOF
clf = BaselineLGB(X, y)
clf.run()
```

## lgb_自定义metric

```python
from sklearn import metrics
def ks(y_hat, data):
    y_true = data.get_label()
    fpr,tpr,thres = metrics.roc_curve(y_true,y_hat,pos_label=1)
    return 'ks', abs(fpr - tpr).max(), True

lgb_data = lgb.Dataset(X, y)

lgb.cv(
    params,
    lgb_data,
    num_boost_round=2000,
    nfold=5,
    stratified=False, # 回归一定是False
    early_stopping_rounds=100,
    verbose_eval=50,
    feval = ks, #ks  #这里增加feval参数
    show_stdv=True)
```

## lgb常用参数
### 1. 原生接口
- 分类
```python
params = {'boosting_type': 'gbdt',# 'rf', 'dart', 'goss'
          'objective': 'binary',# 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
          'max_depth': -1,
          'num_leaves': 63, # 根据具体问题调整
          'learning_rate': 0.01,
          'min_split_gain': 0.0,
          'min_child_weight': 0.001,
          'min_child_samples': 20,
          'subsample': 0.8,
          'subsample_freq': 8,
          'colsample_bytree': 0.8,
          'reg_alpha': 0.0,
          'reg_lambda': 0.0,
          'scale_pos_weight': 1,
          'random_state': None,
          'n_jobs': 32}
```

- 回归
```python
params = {
    'boosting': 'gbdt', # 'rf', 'dart', 'goss'
    'application': 'regression',
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
    'metric': 'rmse',
    'num_threads': 32,
}
```

---
```python
lgb_data = lgb.Dataset(X, y)

lgb.cv(
    params,
    lgb_data,
    num_boost_round=2000,
    nfold=5,
    stratified=False, # 回归一定是False
    metrics=None,
    early_stopping_rounds=50,
    verbose_eval=50,
    show_stdv=True,
    seed=0
)
       
lgb.train(
    params,
    lgb_data,
    num_boost_round=2000,
    valid_sets=None,
    early_stopping_rounds=50,
    verbose_eval=50
)
```
---
### 2. SK接口
- 分类
```python
clf = LGBMClassifier(
    boosting_type='gbdt',  # 'rf', 'dart', 'goss'
    objective='binary',  # objective='multiclass', num_class = 3
    max_depth=-1,
    num_leaves=2 ** 7 - 1,
    learning_rate=0.01,
    n_estimators=1000,

    min_split_gain=0.0,
    min_child_weight=0.001,

    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=0.0,

    scale_pos_weight=1,  # is_unbalance=True 不能同时设

    random_state=888,
    n_jobs=-1
)

```

- 回归
```python
clf = LGBMRegressor(
    boosting_type='gbdt',  # 'rf', 'dart', 'goss'
    objective='regression',
    max_depth=-1,
    num_leaves=2 ** 7 - 1,
    learning_rate=0.01,
    n_estimators=1000,

    min_split_gain=0.0,
    min_child_weight=0.001,

    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,

    reg_alpha=0.0,
    reg_lambda=0.0,

    scale_pos_weight=1,  # is_unbalance=True 不能同时设

    random_state=888,
    n_jobs=-1
)

```

---
```python
clf.fit(
    X_train,
    y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_metric='logloss',
    early_stopping_rounds=100,
    verbose=50,
    feature_name='auto',
    categorical_feature='auto'
)
```

## oof

```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
def train_model(X, y, X_test, cv, cv_seed, lgb_seed):
    params = {'boosting_type': 'gbdt',# 'rf', 'dart', 'goss'
          'objective': 'binary',# 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
          'max_depth': 7,
          'n_estimators': 3000,
          'num_leaves': 63, # 根据具体问题调整
          'learning_rate': 0.02421306293966501,
          'min_split_gain': 0.0,
          'min_child_weight': 19.75111813614344,
          'min_child_samples': 20,
          'subsample': 0.57,
          'subsample_freq': 8,
          'colsample_bytree': 0.1999,
          'reg_alpha': 5.455077557037526,
          'reg_lambda': 5.8700819100907715,
          'scale_pos_weight': 1,
          'random_state': lgb_seed,
          'n_jobs': 32}
    clf = LGBMClassifier(**params)

    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros((X_test.shape[0], cv))
    for n_fold, (train_idx, valid_idx) in enumerate(StratifiedKFold(cv, True, cv_seed).split(X, y), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='auc',
                early_stopping_rounds=300,
                verbose=0)
        oof_preds[valid_idx] = clf.predict_proba(X_valid)[:, 1]
        sub_preds[:, n_fold - 1] = clf.predict_proba(X_test)[:, 1]
    sub_preds = sub_preds.mean(1)
    print('TRIN AUC:', roc_auc_score(y, oof_preds))
    return sub_preds

if __name__ == '__main__':
X = new_features[new_features.label!=-1].drop(['label'],axis=1)
y = new_features[new_features.label!=-1].label
X_test = new_features[new_features.label==-1].drop(['label'],axis=1)
train_model(X.values, y.values, X_test.values, 5, 0, 0)
```





遇到的问题：<br>
如果服务器上直接pip install lightgbm那么跑模型的时候可能会非常慢，解决办法：
git clone 源码，重新编译。[LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#linux)<br>
安装完后然后python setup.py intall<br>
import此包的时候可能会报错；<br>
 - libgomp.so.1，GOMP_4.0不存在问题 [解决链接](https://blog.csdn.net/u010486697/article/details/79156723)
 - OSError: /opt/algor/zhoubin/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found [解决链接](https://www.cnblogs.com/weinyzhou/p/4983306.html)
 