# [CatBoost][1]
- CatBoostClassifier
- CatBoostRegressor

## catboost加载数据，类似lgb.Dataset方式

```python
train_weights = 1 / np.square(y_train)
train_dataset = catboost.Pool(x_train[features], label=y_train, cat_features=[0], weight=train_weights)
```

## 自定义catboost评测函数及使用

```python
import numpy as np
from catboost import CatBoostRegressor

class RMSPE:
    @staticmethod
    def rmspe(y_true, y_pred):
        return np.sqrt(np.mean(np.square((y_true - y_pred) / y_true)))
    def is_max_optimal(self):
        return True
    def evaluate(self, approxes, target, weight):            
        assert len(approxes) == 1
        assert len(target) == len(approxes[0])
        approx = approxes[0]
        score = self.rmspe(target, approx)
        return score, 1
    def get_final_error(self, error, weight):
        return error
        
catb_model = CatBoostRegressor(iterations=3000,
                             learning_rate=0.05,
                             depth=7,
                             subsample=0.7,
                             loss_function='RMSE',
                             eval_metric=RMSPE(),
                             #l2_leaf_reg = 0.001,
                             #random_strength = 0.5,
                             #bagging_temperature=0.8,
                             #task_type="GPU",
                             random_seed=2021,
                             od_type='Iter',
                             metric_period=75,
                             od_wait=100)
catb_model.fit(
    X=train.loc[train_idx, features], y=train.loc[train_idx, 'target'].to_numpy(),
    sample_weight = train.loc[train_idx, 'target_sqr'].to_numpy(),
    eval_set = (train.loc[valid_idx, features], train.loc[valid_idx, 'target'].to_numpy(),),
    early_stopping_rounds = 20,
    cat_features = [0], #第一个特征定为类别特征
    verbose=False)

#sk接口模型保存与读取
catb_model.save_model(f"./catb{fold}.model") #存储
catb_model = catb_model.load_model(f'./catb{fold}.model')#读取
```

## xx









---
[1]: https://tech.yandex.com/catboost/doc/dg/concepts/python-quickstart-docpage/#classification-and-regression
