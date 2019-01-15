rmse for regression and error for classification

这里的情况是有五类打分，评判标准是RMSE（开根号的MSE）<br>
这种情况下，可以采取分类＋回归的方式

数据层面：<br>
数据清洗，构建特征（词向量＋cnn特征＋rnn特征）<br>
算法层面：<br>
实现取决于评判标准而采取不同的策略。比如这里评判标准是RMSE，但training label又是5大类，所以我想希望能够把分类做到极致，再作为回归的制约进行结果确定。

**如何把分类做到极致？**<br>
如果是多分类问题，可以先看下数据类别的分布情况，比如5分占了60%或者5分和4分占了百分之七八十，那我就先把这个问题看成二分类问题，尽量把4或5分的分对那么整个结果就不会太差，1 2 3分的取平均，4和5取平均，这个到时候也可以作为特征加入！

分类完后，再做下回归的融合：<br>
回归可选的模型很多，比如xgb＋lgb＋nb＋lr+svr等融合策略比如取平均，得到最终的分值（选择模型的标准是使每个预测的结果精度尽可能的高同时相关性比如皮尔逊系数尽可能的低）

最后分类和回归再做进一步的融合即可！<br>
比如分类得到3分，回归得到2.6分，那么最终取3分;如果分类得到4分，回归得到1.1分，则取平均等等策略。

**跑模型流程之cv使用 选合适的参数来提高精度**
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold

forest = RandomForestClassifier(n_estimators = 120,max_depth=5, random_state=42)
cross_val_score(forest,X=train_data_features,y=df.Score,scoring='neg_mean_squared_error',cv=3)

# 这里的scoring可以自己写，比如我想用RMSE则
from sklearn.metrics import scorer
def ff(y,y_pred):
    rmse = np.sqrt(sum((y-y_pred)**2)/len(y))
    return rmse
rmse_scoring = scorer.make_scorer(ff)
cross_val_score(forest,X=train_data_features,y=df.Score,scoring=rmse_scoring,cv=3)
```
新手用cross_val_score比较简单，后期可用KFold更灵活，
回归用MSE或RMSE(smaller better)，二分类用auc来评判，多分类用accuracy，cv一般选5折（根据线下训练和线上测试数据进行几折选择，比如train是10000，线上test是3000，那么就选3折）


**xgb是gbdt的优化：主要两方面，**<br>
1.目标函数，传统GBDT在优化时只用到一阶导数信息（负梯度），xgboost则对代价函数进行了二阶泰勒展开，同时用到一阶和二阶导数<br>
2.我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），xgboost在训练之前，预先对数据进行排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。

这个block结构也使得并行成为了可能，在进行节点分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。<br>
（后期用直方图算法，用于高效地生成候选的分割点）

**lgb对xgb的优化**<br>
1.基于Histogram的决策树算法<br>
2.带深度限制的Leaf-wise的叶子生长策略<br>
3.直方图做差加速<br>
4.直接支持类别特征(Categorical Feature)<br>
5.Cache命中率优化<br>
6.基于直方图的稀疏特征优化<br>
7.多线程优化

**跑tensorflow rnn时的坑**<br>

dtype有问题！
![dtype_problem.png](dtype_problem.png)