<h1 align = "center">:alien: Metrics :alien:</h1>

---

### [sklearn.metrics][11]

### 准确率 Accuracy

用的不多<br>
```python
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```

### Kolmogorov-Smirnov(KS)

一般不单独用<br>
[KS](https://www.cnblogs.com/bergus/p/shu-ju-wa-jue-shu-yu-jie-xi.html)

### MSE

回归中使用
```python
# MSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)
```

### Area Under the ROC(AUC) or f1_score

分类中使用，尤其是样本不平衡中
```python
# F1_score
from sklearn.metrics import f1_score
f1_score(y_true, y_pred)
```
```python
# AUC
import numpy as np
from sklearn import metrics
y_true = np.array([1, 1, 2, 2])
y_pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
metrics.auc(fpr, tpr)
#0.75
```
---

 [0]: https://github.com/benhamner/Metrics/tree/master/Python
 [1]: http://img.blog.csdn.net/20150924153157802
 [2]: http://third.datacastle.cn/pkbigdata/master.other.img/7372d308-8d38-4e45-8ab7-ffab7763096a.png
 [3]: https://github.com/Jie-Yuan/DataMining/raw/master/7_Metrics/Pictures/11.png
 [11]: http://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values
 [12]: http://www.cnblogs.com/harvey888/p/6964741.html
 [13]: https://img-blog.csdn.net/20171012171557401?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzQyMTYyOQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast
 [14]: https://www.cnblogs.com/bergus/p/shu-ju-wa-jue-shu-yu-jie-xi.html
 [15]: https://mp.weixin.qq.com/s?__biz=MzI1MzY0MzE4Mg==&mid=2247483981&idx=1&sn=f347a44a7b41693bc923de91d159dbf3&chksm=e9d0128cdea79b9a5628e932f614a681867d76307a90a0db67e8195c0b855e6a5160528a72d0&scene=0#rd
