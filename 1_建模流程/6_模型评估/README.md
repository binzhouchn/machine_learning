<h1 align = "center">:helicopter: Metrics :running:</h1>

---

### [sklearn.metrics][11]

### 1. 准确率 Accuracy

用的不多<br>
```python
# Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_true, y_pred)
```

### 2. Kolmogorov-Smirnov(KS)

一般不单独用<br>
评分卡用LR建模，如果评价指标是KS则要把pred_prob转成pred_log_prob<br>
[KS](https://www.cnblogs.com/bergus/p/shu-ju-wa-jue-shu-yu-jie-xi.html)

### 3. MSE

回归中使用
```python
# MSE
from sklearn.metrics import mean_squared_error
mean_squared_error(y_true, y_pred)
```

### 4. Area Under the Curve(AUC) or f1_score

![](auc_calc.png)<br>
AUC计算示例<br>
如上图所示，我们有8个测试样本，模型的预测值（按大小排序）和样本的真实标签如右表所示，绘制ROC曲线的整个过程如下所示：<br>
1、令阈值等于第一个预测值0.91，所有大于等于0.91的预测值都被判定为阳性，此时TPR=1/4，FPR=0/4，所有我们有了第一个点（0.0，0.25）<br>
2、令阈值等于第二个预测值0.85，所有大于等于0.85的预测值都被判定为阳性，这种情况下第二个样本属于被错误预测为阳性的阴性样本，也就是FP，所以TPR=1/4，FPR=1/4，所以我们有了第二个点（0.25，0.25）<br>
3、按照这种方法依次取第三、四...个预测值作为阈值，就能依次得到ROC曲线上的坐标点（0.5，0.25）、（0.75，0.25）...（1.0，1.0）<br>
4、将各个点依次连接起来，就得到了如图所示的ROC曲线<br>
5、计算ROC曲线下方的面积为0.75，即AUC=0.75<br>

[参考](https://baijiahao.baidu.com/s?id=1597939133517926460&wfr=spider&for=pc)

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
y_pred = np.array([0.1, 0.4, 0.35, 0.8]) # [1,4,3.5,8]出来的auc值是一样的
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
metrics.auc(fpr, tpr)
#0.75
# 或者
from sklearn.metrics import roc_auc_score
roc_auc_score(y_true, y_pred)
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
