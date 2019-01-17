[<h1 align = "center">:helicopter: 集成学习基本策略 :running:</h1>][0]

---

## 1. Bagging（Booststrap AGGregatING）

 - **基于并行策略**：基学习器之间不存在依赖关系，可同时生成
 - **基本思路:**
    - 利用`自助采样法`对训练集随机采样，重复进行 T 次；
    - 基于每个采样集训练一个基学习器，并得到 T 个基学习器；
    - 预测时，集体**投票决策****
        > `自助采样法`：对 m 个样本的训练集，有放回的采样 m 次；此时，样本在 m 次采样中始终没被采样的概率约为
         0.368，即每次自助采样只能采样到全部样本的 63% 左右<br>
         ![pic1](pic/公式_20180902220459.png)
 - **特点：**
    - 训练每个基学习器时只使用一部分样本；
    - 偏好`不稳定的学习器`作为基学习器；所谓`不稳定的学习器`，指的是对样本分布较为敏感的学习器。

## 2. Boosting

 - **基于串行策略**
 
基学习器之间存在依赖关系，新的学习器需要根据上一个学习器生成<br>
其主要思想是将弱分类器组装成一个强分类器。在PAC（概率近似正确）学习框架下，则一定可以将弱分类器组装成一个强分类器。

 - **基本思路**
    - 先从初始训练集训练一个基学习器；初始训练集中各样本的权重是相同的；
    - 根据上一个基学习器的表现，调整样本权重，使分类错误的样本得到更多的关注；
    - 基于调整后的样本分布，训练下一个基学习器；
    - 测试时，对各基学习器加权得到最终结果
 
 - **特点**：每次学习都会使用全部训练样本
 - **代表算法：**
    - [AdaBoost算法](https://blog.csdn.net/guyuealian/article/details/70995333)
    - [GBDT算法](http://www.jianshu.com/p/005a4e6ac775)
    - xgboost
    - lgb

关于Boosting的两个核心问题：

1. 在每一轮如何改变训练数据的权值或概率分布？

通过提高那些在前一轮被弱分类器分错样例的权值，减小前一轮分对样例的权值，来使得分类器对误分的数据有较好的效果。

2. 通过什么方式来组合弱分类器？

通过加法模型将弱分类器进行线性组合，比如AdaBoost通过加权多数表决的方式（使用加权的投票机制代替平均投票机制），即增大错误率小的分类器的权值，同时减小错误率较大的分类器的权值。而提升树通过拟合残差的方式逐步减小残差，将每一步生成的模型叠加得到最终模型。




## 3. Stacking

stacking经典图<br>
![stacking经典图](stacking1.png)

![stacking2](stacking2.png)

上图展示了使用5-Fold进行一次Stacking的过程<br>
主要步骤是，比如数据是200个特征，样本数是10万个，
base model经过5折cv(一般业界一折就行)以后得到10万个预测值（即生成一个新特征）<br>
多个基模型就有了多个特征，最后再跑一个meta模型

```python
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
class StackingAveragedModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
   
    # We again fit the data on clones of the original models
    # 三个base model得到的就是三个new features，然后再和y跑一个meta model
    def fit(self, X, y):
        self.base_models_ = [list() for x in self.base_models]
        self.meta_model_ = clone(self.meta_model)
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=156)
        
        # Train cloned base models then create out-of-fold predictions
        # that are needed to train the cloned meta-model
        out_of_fold_predictions = np.zeros((X.shape[0], len(self.base_models)))
        for i, model in enumerate(self.base_models):
            for train_index, holdout_index in kfold.split(X, y):
                instance = clone(model)
                self.base_models_[i].append(instance)
                instance.fit(X[train_index], y[train_index])
                y_pred = instance.predict(X[holdout_index])
                out_of_fold_predictions[holdout_index, i] = y_pred
                
        # Now train the cloned  meta-model using the out-of-fold predictions as new feature
        self.meta_model_.fit(out_of_fold_predictions, y)
        return self
   
    #Do the predictions of all base models on the test data and use the averaged predictions as 
    #meta-features for the final prediction which is done by the meta-model
    def predict(self, X):
        meta_features = np.column_stack([
            np.column_stack([model.predict(X) for model in base_models]).mean(axis=1)
            for base_models in self.base_models_ ])
        return self.meta_model_.predict(meta_features)

# 回归
stacked_averaged_models = StackingAveragedModels(base_models = (ENet, model_xgb, model_lgb),
                                                 meta_model = lasso)
stacked_averaged_models.fit(X.values, y)
```

## 3. Blending

Blending与Stacking类似，但单独留出一部分数据（如20%）用于训练Stage X模型

## 4. Bagging Ensemble Selection

Bagging Ensemble Selection在CrowdFlower搜索相关性比赛中使用的方法，其主要的优点在于可以以优化任意的指标来进行模型集成。
这些指标可以是可导的（如LogLoss等）和不可导的（如正确率，AUC，Quadratic Weighted Kappa等）。它是一个前向贪婪算法，存在过拟合的可能性，
作者在文献中提出了一系列的方法（如Bagging）来降低这种风险，稳定集成模型的性能。使用这个方法，需要有成百上千的基础模型。
为此，在CrowdFlower的比赛中，调参过程中所有的中间模型以及相应的预测结果保留下来，作为基础模型。这样做的好处是，
不仅仅能够找到最优的单模型（Best Single Model），而且所有的中间模型还可以参与模型集成，进一步提升效果。

---
## 5. 多样性
- 误差——分歧分解
- 多样性度量
- 多样性增强
    - 数据样本扰动
    - 输入属性扰动
    - 算法参数扰动
    - 输出表示扰动
        - 翻转法(Flipping Output)：随机改变一些训练样本标记
        - 输出调制法(Output Smearing)：分类输出转化为回归输出
        - OVO/ECOC



---

[0]: http://www.cnblogs.com/jasonfreak/p/5657196.html
