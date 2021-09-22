<h1 align = "center">:rocket: 自定义评估函数 :facepunch:</h1>

---
## Xgb

```python
def feval(y_pred, y_true):
    from ml_metrics import auc
    y_true = y_true.get_label()
    return '-auc', - auc(y_true, y_pred)
    
XGBClassifier().fit(X, y, eval_metric=feval) # 满足目标最小化
xgb.train(feval=feval, maximize=False) # 目标最大化可选
xgb.cv(feval=feval, maximize=False) # 目标最大化可选
```

---
## Lgb
```python
def feval(y_pred, y_true):
    from ml_metrics import auc
    y_true = y_true.get_label()
    return 'auc', auc(y_true, y_pred), True# maximize=False比xgb多返回一项

# LGBMClassifier().fit(X, y,eval_metric=feval) # 待实验
lgb.train(feval=feval)
lgb.cv(feval=feval)
```

---
##
```python
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
cross_val_score(scoring) # 自定义评估函数需要make_scorer包装一下
```
---
https://www.cnblogs.com/silence-gtx/p/5812012.html
