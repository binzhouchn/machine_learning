<h1 align = "center">:helicopter: 提升算法 :running:</h1>

---
## Parameters
- [Xgboost][1]
- [LightGBM][2]
- [GBDT-PL][7]
- [LinXGBoost][8]

 http://dwz.cn/74863J
---
## 常用参数速查
|[**xgb**][3]|[**lgb**][5]|[**xgb.sklearn**][4]|[**lgb.sklearn**][6]|
|:--|:--|:--|:--|
|booster='gbtree'|boosting='gbdt'|booster='gbtree'|boosting_type='gbdt'|
|objective='binary:logistic'|application='binary'|objective='binary:logistic'|objective='binary'|
|max_depth=7|num_leaves=2**7|max_depth=7|num_leaves=2**7|
|eta=0.1|learning_rate=0.1|learning_rate=0.1|learning_rate=0.1|
|num_boost_round=10|num_boost_round=10|n_estimators=10|n_estimators=10|
|gamma=0|min_split_gain=0.0|gamma=0|min_split_gain=0.0|
|min_child_weight=5|min_child_weight=5|min_child_weight=5|min_child_weight=5|
|subsample=1|bagging_fraction=1|subsample=1.0|subsample=1.0|
|colsample_bytree=1.0|feature_fraction=1|colsample_bytree=1.0|colsample_bytree=1.0|
|alpha=0|lambda_l1=0|reg_alpha=0.0|reg_alpha=0.0|
|lambda=1|lambda_l2=0|reg_lambda=1|reg_lambda=0.0|
|scale_pos_weight=1|scale_pos_weight=1|scale_pos_weight=1|scale_pos_weight=1|
|seed |bagging_seed<br/>feature_fraction_seed|random_state=888|random_state=888|
|nthread|num_threads|n_jobs=4|n_jobs=4|
|evals|valid_sets|eval_set|eval_set|
|eval_metric|metric|eval_metric|eval_metric|
|early_stopping_rounds|early_stopping_rounds|early_stopping_rounds|early_stopping_rounds|
|verbose_eval|verbose_eval|verbose|verbose|


## oof用法

**0，1二分类cv**<br>
```python
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score
from tqdm import tqdm
def train_model(X, y, X_test, cv, cv_seed, lgb_seed):
    params = {} # 参数详情具体见1_lgb/oof.py
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

**多分类cv**<br>
```python
test = tf_feat[train_shape[0]:] 

kf = StratifiedKFold(n_splits=N,random_state=42,shuffle=True)
oof = np.zeros((X.shape[0],3))
oof_sub = np.zeros((sub.shape[0],3))
for j,(train_in,dev_in) in enumerate(kf.split(X,y)):
    print('running',j)
    X_train,X_dev,y_train,y_dev = X[train_in],X[dev_in],y[train_in],y[dev_in]
    clf = LogisticRegression(C=4)
    clf.fit(X_train,y_train)
    dev_y = clf.predict_proba(X_dev)
    oof[dev_in] = dev_y
    oof_sub = oof_sub + clf.predict_proba(test)

xx_cv = f1_score(y,np.argmax(oof,axis=1),average='macro')
print(xx_cv)
```


---
[1]: http://xgboost.readthedocs.io/en/latest/parameter.html#
[2]: https://lightgbm.readthedocs.io/en/latest/Parameters.html#
[3]: https://github.com/Jie-Yuan/DataMining/blob/master/5_PopularAlgorithm/1_Boosting/2_xgb/README.md#1-%E5%8E%9F%E7%94%9F%E6%8E%A5%E5%8F%A3
[4]: https://github.com/Jie-Yuan/DataMining/blob/master/5_PopularAlgorithm/1_Boosting/2_xgb/README.md#2-sk%E6%8E%A5%E5%8F%A3
[5]: https://github.com/Jie-Yuan/DataMining/blob/master/5_PopularAlgorithm/1_Boosting/1_lgb/README.md#1-%E5%8E%9F%E7%94%9F%E6%8E%A5%E5%8F%A3
[6]: https://github.com/Jie-Yuan/DataMining/blob/master/5_PopularAlgorithm/1_Boosting/1_lgb/README.md#2-sk%E6%8E%A5%E5%8F%A3
[7]: https://github.com/GBDT-PL/GBDT-PL
[8]: https://github.com/ldv1/LinXGBoost
