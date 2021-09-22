# 目录

[**1. xgboost优化**](#xgboost优化)

[**2. xgboost参数与跑模型示例**](#xgboost参数与跑模型示例)

[**3. xgboost sklearn框架和原生态框架**](#xgboost_sklearn框架和原生态框架)

[**4. xgb调参指南及示例代码**](#xgb调参指南及示例代码)

---

## xgboost优化

**xgb是gbdt的优化：主要两方面**
1.目标函数，传统GBDT在优化时只用到一阶导数信息（负梯度），xgboost则对代价函数进行了二阶泰勒展开，同时用到一阶和二阶导数<br>
2.我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。<br>

这个block结构也使得并行成为了可能，在进行节点分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。 （后期用直方图算法，用于高效地生成候选的分割点）

**lgb对xgb的优化**
1.基于Histogram的决策树算法<br>
2.带深度限制的Leaf-wise的叶子生长策略<br>
3.直方图做差加速<br>
4.直接支持类别特征(Categorical Feature)<br>
5.Cache命中率优化<br>
6.基于直方图的稀疏特征优化<br>
7.多线程优化<br>

## xgboost参数与跑模型示例

```python
import xgboost as xgb

params={
    'booster':'gbtree',
    'objective': 'binary:logistic',
    'scale_pos_weight': 1/7.5,
    #7183正样本
    #55596条总样本
    #差不多1:7.7这样子
    'gamma':0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
    'max_depth':8, # 构建树的深度，越大越容易过拟合
    'lambda':3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
    'subsample':0.7, # 随机采样训练样本
    #'colsample_bytree':0.7, # 生成树时进行的列采样
    'min_child_weight':3, 
    # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
    'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
    'eta': 0.03, # 如同学习率
    'seed':1000,
    'nthread':12,# cpu 线程数
    'eval_metric':'auc',
    'missing':-1
}
plst = list(params.items())
num_rounds = 2000 # 迭代次数
xgb_train = xgb.DMatrix(X, label=y)
xgb_val = xgb.DMatrix(val_X,label=val_y)
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]
model = xgb.train(plst, xgb_train, num_boost_round=75000,evals=watchlist,early_stopping_rounds=500)
```
```python
params = {
        'colsample_bytree': 0.5041920450812235,
        'gamma': 0.690363148214239,
        'learning_rate': 0.01,
        'max_depth': 8,
        'min_child_weight': 9,
        'nthread': 1,
        'objective': 'binary:logistic',
        'reg_alpha': 4.620727573976632,
        'reg_lambda': 1.9231173132006631,
        'scale_pos_weight': 5,
        'seed': 2017,
        'subsample': 0.5463188675095159
        }
```

## xgboost_sklearn框架和原生态框架

### 1. 原生接口

 - 1.1 分类祖传参数
 
```python
"""
max_delta_step: 类别不平衡有助于逻辑回归
"""
params = {
    'booster': 'gbtree', #  'dart' # 'rank:pairwise'对排序友好
   'objective': 'binary:logistic', # 'objective': 'multi:softmax', 'num_class': 3,
    'eta': 0.01,
    'max_depth': 7,

    'subsample': 0.8,
    'colsample_bytree': 0.4,
    'min_child_weight': 10,

    'gamma': 2,
    'eval_metric': 'auc',
    'nthread': 16,
    'seed': 888,
}
```

 - 2.1 回归祖传参数
 
```python
params = {
    'booster': 'gbtree', # 'dart', 'gblinear' 
    'objective': 'reg:linear', # 'reg:tweedie', 'reg:gamma'
    'eta': 0.1,
    'max_depth': 7,

    'gamma': 0,
    'min_child_weight': 1,

    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'alpha': 0,
    'lambda': 1,

    'scale_pos_weight': 1,
    'eval_metric': 'rmse',
    'nthread': 16,
    'seed': 888
}
```

 - 2.2 回归(kaggle optiver比赛参数)
 
[notebook链接](https://www.kaggle.com/binzhouchn/latest-code9-lgb-xgb-catboost)


---
```python
xgb_data = xgb.DMatrix(X, y)

xgb.cv(
    params,
    xgb_data,
    num_boost_round=2000,
    nfold=3,
    stratified=True, # stratified=False # 回归
    metrics=(),
    early_stopping_rounds=50,
    verbose_eval=50,
    show_stdv=True,
    seed=0
)
       
xgb.train(
    params,
    dtrain,
    num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50,
    verbose_eval=50
)
```


---

### 2. SK接口(archive)
- 分类
```python
clf = XGBClassifier(
    booster='gbtree', #  'dart' # 'rank:pairwise'对排序友好
    objective='binary:logistic',  # 'multi:softmax', 
    max_depth=7,
    learning_rate=0.1,
    n_estimators=100,

    gamma=0,
    min_child_weight=1,

    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,

    reg_alpha=0,
    reg_lambda=1,

    scale_pos_weight=1,

    random_state=888,
    n_jobs=-1
)
```
- 回归
```python
clf = XGBRegressor(
    booster='gbtree', # 'dart', 'gblinear' 
    objective='reg:linear', # 'reg:tweedie', 'reg:gamma'
    max_depth=7,
    learning_rate=0.1,
    n_estimators=100,

    gamma=0,
    min_child_weight=1,

    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,

    reg_alpha=0,
    reg_lambda=1,

    scale_pos_weight=1,

    random_state=888,
    n_jobs=-1
)
```
---
```python
clf.fit(
    X_train, 
    y_train,
    sample_weight=None,  # 可初始化样本权重
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='auc',
    early_stopping_rounds=None,
    verbose=50
)
```

## xgb调参指南及示例代码

[xgboost调参指南](https://m.aliyun.com/yunqi/articles/326853?spm=5176.11156381.0.0.d2WovP)

[xgboost调参示例代码](xgboost调参示例代码.py)



---

## 参数 
http://www.cnblogs.com/ljygoodgoodstudydaydayup/p/6665239.html
http://blog.csdn.net/han_xiaoyang/article/details/52665396
