<h1 align = "center">:helicopter: lgb常用参数 :running:</h1>

# 目录

[**1. lgb baseline模型**](#lgb_baseline模型)

[**2. lgb 自定义metric**](#lgb_自定义metric)

[**3. lgb常用参数**](#lgb常用参数)

[**4. lgb参数调优**](#lgb参数调优)

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
#自定义样例一
import numpy as np
from sklearn import metrics
def ks(y_hat, data):
    y_true = data.get_label()
    fpr, tpr, thres = metrics.roc_curve(y_true,y_hat,pos_label=1)
    return 'ks', abs(fpr - tpr).max(), True
#自定义样例二
def rmspe(y_true, y_pred):
    return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
def feval_rmspe(y_pred, lgb_train):
    y_true = lgb_train.get_label()
    return 'RMSPE', rmspe(y_true, y_pred), False

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
 
 - 1.1 分类祖传参数
 
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

 - 2.1 回归祖传参数
 
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

 - 2.2 回归(kaggle optiver比赛参数)
 
[notebook链接](https://www.kaggle.com/binzhouchn/latest-code9-lgb-xgb-catboost)

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

### 2. SK接口(archive)
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

## lgb参数调优

```python
num_folds = 5
group_fold = GroupKFold( n_splits = num_folds )
def objective(trial):

    # Optuna suggest params
    seed = 1111
    params = {
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.2),        
        'lambda_l1': trial.suggest_float('lambda_l1', 0.1, 1),
        'lambda_l2': trial.suggest_float('lambda_l2', 0.1, 1),
        'num_leaves': trial.suggest_int('num_leaves', 400, 800),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.50, 1),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.50, 1),
        'bagging_freq': trial.suggest_int('bagging_freq',1, 2),
        'min_data_in_leaf':trial.suggest_int('min_data_in_leaf',400,700),
        'max_depth': trial.suggest_int('max_depth', 6 , 13),
        'seed': seed,
        'objective': 'rmse',
        'boosting': 'gbdt',
        'verbosity': -1,
        'n_jobs': -1,
        #'device': 'gpu','gpu_platform_id': 0,
        #'gpu_device_id': 0
    }  

    
    rmspe_list = []

    for fold, (trn_ind, val_ind) in enumerate(group_fold.split(X, y, groups = X['time_id'])):
        print(f'Training fold {fold + 1}')
        x_train, x_val = X.iloc[trn_ind], X.iloc[val_ind]
        y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
        # Root mean squared percentage error weights
        train_weights = 1 / np.square(y_train)
        val_weights = 1 / np.square(y_val)
        train_dataset = lgb.Dataset(x_train, y_train, weight = train_weights)
        val_dataset = lgb.Dataset(x_val, y_val, weight = val_weights)
        model = lgb.train(params = params,
                          num_boost_round=5000,
                          train_set = train_dataset, 
                          valid_sets = [train_dataset, val_dataset], 
                          verbose_eval = 250,
                          early_stopping_rounds=50,
                          feval = feval_rmspe)
        
        #preds = model.predict(d_val)
        preds = model.predict(x_val[features])
        score = rmspe(y_val, preds)
        rmspe_list.append(score)
    
    print(f'Trial done: rmspe values on folds: {score}')
    return np.mean(rmspe_list)
    
n_trials = 10

FIT_LGB = True

if FIT_LGB:
    study = optuna.create_study(direction="minimize",study_name = 'LGB')
    study.optimize(objective)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
```






---

遇到的问题：<br>
如果服务器上直接pip install lightgbm那么跑模型的时候可能会非常慢，解决办法：
git clone 源码，重新编译。[LightGBM](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html#linux)<br>
安装完后然后python setup.py intall<br>
import此包的时候可能会报错；<br>
 - libgomp.so.1，GOMP_4.0不存在问题 [解决链接](https://blog.csdn.net/u010486697/article/details/79156723)
 - OSError: /opt/algor/zhoubin/anaconda3/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.20' not found [解决链接](https://www.cnblogs.com/weinyzhou/p/4983306.html)
 