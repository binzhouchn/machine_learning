from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score

params = {
    'boosting': 'gbdt', # 'rf', 'dart', 'goss'
    'application': 'binary', # 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
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
#     'metric': 'auc,
    'num_threads': 32,
}

# 这里可以用自带的auc参数，但是我们想要用自定义的怎么办呢，以下是解决方案(自定义函数参照这个写就行了)
# 参考 https://lightgbm.readthedocs.io/en/latest/Python-API.html
def lgb_auc(y_hat, data): # 输入一定是y_pred，和data
    y_true = data.get_label()
    y_true = y_true + 1
    fpr, tpr, thresholds = roc_curve(y_true, y_hat, pos_label=2)
    return 'auc', auc(fpr, tpr), True

# 自定义f1_score
def lgb_f1_score(y_hat, data):
    y_true = data.get_label()
#     y_hat = np.round(y_hat)
    y_hat = [1 if x > 0.3 else 0 for x in y_hat]
    return 'f1_score', f1_score(y_true, y_hat), True

lgb_data = lgb.Dataset(X, y)

lgb.cv(
    params,
    lgb_data,
    num_boost_round=2000,
    nfold=5,
    stratified=False, # 回归一定是False
    early_stopping_rounds=100,
    verbose_eval=50,
    feval = lgb_auc, #lgb_f1_score # 这里增加feval参数
    show_stdv=True)
