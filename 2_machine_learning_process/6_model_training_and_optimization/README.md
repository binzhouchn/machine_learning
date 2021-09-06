<h1 align = "left">:helicopter: 模型训练和验证 :running:</h1>

---

[各种算法优缺点1](https://mp.weixin.qq.com/s?__biz=MzA4OTg5NzY3NA==&mid=2649345665&idx=1&sn=000c6e1ceada252162b803404d9a397c&chksm=880e8124bf790832dfc5b10e142425969799639743295078ee1d9524ab21e7ad1b314136d923&mpshare=1&scene=1&srcid=0528p1yaSx6dNlRh0U58XebG#rd)<br>
[各种算法优缺点2](https://mp.weixin.qq.com/s/6hD19wWEex-0s-dweuP5sg)<br>
[各种算法优缺点3](https://blog.csdn.net/u012422446/article/details/53034260)<br>

# 1. 数据训练oof

参考地址：https://www.kaggle.com/binzhouchn/latest-code2?scriptVersionId=73995696

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import numpy as np
'''
train(这里train，test是一个dataframe)
test
y = train['target']
'''
oof_predictions = np.zeros(train.shape[0])
test_predictions = np.zeros(test.shape[0])
# Create a KFold object
kfold = KFold(n_splits = 5, random_state = 2021, shuffle = True)
# Iterate through each fold
for fold, (trn_ind, val_ind) in enumerate(kfold.split(train)):
    print(f'Training fold {fold + 1}')
    x_train, x_val = train.iloc[trn_ind], train.iloc[val_ind]
    y_train, y_val = y.iloc[trn_ind], y.iloc[val_ind]
    # Root mean squared percentage error weights
    train_weights = 1 / np.square(y_train)
    val_weights = 1 / np.square(y_val)
    train_dataset = lgb.Dataset(x_train[features], y_train, weight = train_weights)
    val_dataset = lgb.Dataset(x_val[features], y_val, weight = val_weights)
    model = lgb.train(params = params,
                      num_boost_round=1300,
                      train_set = train_dataset, 
                      valid_sets = [train_dataset, val_dataset], 
                      verbose_eval = 250,
                      early_stopping_rounds=50, #50
                      feval = feval_rmspe)
    # Add predictions to the out of folds array
    oof_predictions[val_ind] = model.predict(x_val[features])
    # Predict the test set
    test_predictions += model.predict(test[features]) / 5
rmspe_score = rmspe(y, oof_predictions)

```

# 2. 经验参数

![经验参数](经验参数.jpg)

# 3. 模型选择

 - 对于稀疏型特征（如文本特征，One-hot的ID类特征），我们一般使用线性模型，譬如 Linear Regression 或者 Logistic Regression。Random Forest 和 GBDT 等树模型不太适用于稀疏的特征，但可以先对特征进行降维（如PCA，SVD/LSA等），再使用这些特征。稀疏特征直接输入 DNN 会导致网络 weight 较多，不利于优化，也可以考虑先降维，或者对 ID 类特征使用 Embedding 的方式
 
 - 对于稠密型特征，推荐使用XGBoost进行建模，简单易用效果好
 
 - 数据中既有稀疏特征，又有稠密特征，可以考虑使用线性模型对稀疏特征进行建模，将其输出与稠密特征一起再输入XGBoost/DNN建模，具体可以参考5_模型集成中Stacking部分

## 3.1 分类

```python
import lightgbm as lgb

lgb_data = lgb.Dataset(X, y)
params = {
    'boosting': 'gbdt', # 'rf', 'dart', 'goss'
    'application': 'binary', # 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
    'learning_rate': 0.01,
    'max_depth': -1,
    'num_leaves': 2 ** 7 - 1, # 根据具体问题调整
    
    'max_bin':255,
    'metric_freq':10,
    
    'min_split_gain': 0,
    'min_child_weight': 1,

    'bagging_fraction': 0.8,
    'feature_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 1000,
    'min_sum_hessian_in_leaf': 5.0,
    'lambda_l1': 0,
    'lambda_l2': 1,

    'scale_pos_weight': 1,
    'metric': 'auc',
    'num_threads': 32,
}
# 做cv
lgb.cv(
    params,
    lgb_data,
    num_boost_round=2000,
    nfold=5,
    stratified=False, # 回归一定是False
    early_stopping_rounds=100,
    verbose_eval=50,
    show_stdv=True)
# 跑模型
lgb.train(
    params,
    lgb_data,
    num_boost_round=2000,
    early_stopping_rounds=100,
    verbose_eval=50)
```

## 3.2 回归

正则化方法（Lasso回归，岭回归和ElasticNet）在数据集中的变量之间
具有高纬度和多重共线性的情况下也能有良好的效果。

3.2.1 参考代码一<br>
```python
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb

# Define a cross validation strategy
#Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return rmse
    
# base models

# This model may be very sensitive to outliers. So we need to made it more robust on them. For that we use the sklearn's Robustscaler() method on pipeline
lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0005, random_state=1)) 
ENet = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
GBoost = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
model_lgb = lgb.LGBMRegressor(objective='regression',num_leaves=5,
                              learning_rate=0.05, n_estimators=720,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2319,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
                                                                                                                                 
# 跑模型看下分数
score = rmsle_cv(models_)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))
```

3.2.2 参考代码二<br>
```python
num_round=600
param={
        'max_depth':2,
        'eta':0.1,
        'gamma':0,
        'min_child_weight':0,
        'save_period':0,
        'booster':'gbtree',
        'silent':1,
        "seed": 0,
        #'subsample':0.7,
        #'colsample_bytree':0.9,
        #'colsample_bylevel':0.9,
        'lambda':6.0,
        'alpha':4.0,
        'objective':'reg:linear',
        #'objective':'count:poisson'
    }

log_num_round=81
log_param={
    'max_depth':5,
    'eta':0.08,
    'gamma':0,
    'min_child_weight':0,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':7.0,
    'alpha':7.0,
    'objective':'reg:linear',
    #'objective':'count:poisson'
}

sqrt_num_round=321
sqrt_param={
    'max_depth':3,
    'eta':0.08,
    'gamma':0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':7.0,
    'alpha':3.0,
    'objective':'reg:linear',
    #'objective':'count:poisson'
}

pow_num_round=20 #172
pow_param={
    'max_depth':3,
    'eta':0.08,
    'gamma':0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':5.0,
    #'alpha':1.0,
    'objective':'reg:linear',
    #'objective':'count:poisson'
}

poi_num_round=607
poi_param={
    'max_depth':2,
    'eta':0.2,
    'gamma':0,
    'min_child_weight':1,
    'save_period':0,
    'booster':'gbtree',
    'silent':1,
    "seed": 0,
    #'subsample':0.7,
    #'colsample_bytree':0.9,
    #'colsample_bylevel':0.9,
    'lambda':2.8,
    #'alpha':3.0,
    #'objective':'reg:linear',
    'objective':'count:poisson'
}
data_length=len(y_data)
num_fold=3
def rmse(pred,real):
    '''
    rmse计算函数
    '''
    #pred=np.round(pred)
    l=len(real)
    rmse_sum=0.0
    for i in range(l):
        rmse_sum+=np.square(pred[i]-real[i])
    return np.sqrt(rmse_sum/l)

def train_pred(X_test,y_data,X_data,param,num_round):
    '''
    通过交叉验证进行调参，得到效果较好的模型
    返回值：训练数据的预测值，真实值，测试数据的预测值
    '''
    k_fold=KFold(num_fold,shuffle=True,random_state=20)
    pred=[]
    real=[]
    model=[]
    dtest=xgb.DMatrix(X_test)
    test_score=[]
    for train,valid in k_fold.split(X_data,y_data):
        X_train=X_data[train]
        y_train=y_data[train]
        X_valid=X_data[valid]
        y_valid=y_data[valid]

        dtrain=xgb.DMatrix(X_train,label=y_train)
        dvalid=xgb.DMatrix(X_valid,label=y_valid)
        watchlist=[(dvalid,'eval'),(dtrain,'train')]
        bst=xgb.train(param,dtrain,num_round,watchlist,learning_rates=None)
        test_score.append(bst.predict(dtest))
        valid_pred=bst.predict(dvalid)
        valid_real=y_valid
        pred.append(valid_pred)
        real.append(valid_real)
        #input('跑完一圈！回车继续...')
    
    return pred,real,test_score

#原始数据
rmse_cross=[]
pred,real,test_results=train_pred(X_test,y_data,X_data,param,num_round)
#泊松回归
rmse_cross=[]
pred,real,poi_test_result=train_pred(X_test,y_data,X_data,poi_param,poi_num_round)
#对score取对数后再训练
#对score开方后再训练
#对score平方后再训练

# 这里原始数据和泊松回归跑出来的结果最好，所以最后就用了这两个模型的预测结果取平均作为预测值。
test_results.extend(poi_test_result) # 两个结果拼接起来(list方法)
test_results=np.array(test_results).mean(axis=0)
```

# 4. 调参和模型验证

 - 训练集和验证集的划分
 
 - 指定参数空间
 
 - 按照一定的方法进行参数搜索

[具体展开看此链接](https://m.sohu.com/a/139981834_116235)

