# 面试中常见的问题:running:

## 1. 聊聊项目

**1.1 最熟悉的项目（描述一下流程最全的项目 催收bot）**

**1.2 金融风控中评分卡建模过程（以上补充）**

[1.1和1.2后期合并一下](../3_1_financial_risk/评分卡算法建模流程.md)

**1.3 知识图谱（建图过程）**

**1.4 智能客服**

[智能客服项目流程](../3_3_nlp_text/.机器人流程及遇到的问题.md)

**1.5 催收机器人**

---

## 2. 项目workflow

 - 确定业务场景及目标
 - 理解实际问题，抽象为机器学习能处理的数学问题
    - 明确可以获得的数据
    - 机器学习的目标是分类、回归还是聚类
    - 算法评价指标：比如AUC, RMSE
 - 创建common sense baseline
 - 获取数据、分割数据(optional)
 - EDA
 - 预处理
 - 特征工程
 - 模型开发
 - 模型集成
 - 模型部署
 - 模型监控
 - 迭代

## 3. 经典的优化方法
 
[梯度下降法](https://zhuanlan.zhihu.com/p/36564434)
选择每一步的方向：泰勒公式！

 - 牛顿法和拟牛顿法
 - 共轭梯度法(Adagrad, Adadelta, RMSprop, Adam)

## 4. 推导SVM算法

最大间隔超平面背后的原理：
 - 相当于在最小化权重时对训练误差进行了约束——对比 L2 范数正则化，则是在最小化训练误差时，对特征权重进行约束
 - 相当于限制了模型复杂度——在一定程度上防止过拟合，具有更强的泛化能力

推导看统计学习方法P100

svm损失函数：[合页损失(加正则项)](https://www.jianshu.com/p/fe14cd066077)


## 5. 推导LR算法

顺便可以看下线性回归，损失函数：均方误差MSE

看统计学习方法P78及<br>
[LR推导及代码实现（推荐）](https://zhuanlan.zhihu.com/p/36670444)

LR损失函数：对数损失logloss(逻辑回归采用对数损失函数来求参数，实际上与采用极大似然估计来求参数是一致的)

## 6. 看下决策树相关的知识点

西瓜书 4.1/4.2/4.3/4.4 <br>
统计学习方法 第五章

信息增益：得知特征X的信息而使得类Y的信息的不确定性减少的程度<br>
g(D,A) = H(D)-H(D|A) 具体公式含义及计算方式看统计学习方法p61,p62<br>
【缺点】以信息增益作为划分训练数据集的特征，存在偏向于选择取值较多的特征的问题

信息增益比（改进信息增益的缺点）：g(D,A) / Ha(D)

ID3用的是信息增益<br>
C4.5用的是信息增益比<br>
CART：回归MSE；分类Gini，gini系数越小越好

[cart剪枝](https://www.zhihu.com/question/22697086)

注意看下ID3, C4.5/C5.0的属性都要是离散的，CART的属性可以是连续的，
而且cart是回归树，后面用的默认的rf, xgb, lgb用的都是回归树

Bagging方法：Random Forest

Random Forest：
 - 样本扰动和属性扰动(k=logd)
 - 降低方差
 - 预测：n棵树投票

Boosting方法：Adaboost, GBDT, xgb, lgb

[Adaboost](https://blog.csdn.net/guyuealian/article/details/70995333)
 - 基本思想：调整样本权重
 - 基(弱)分类器选取（一般单层决策树或基于其他算法）
 - 误差率即每个样本的权值中分类错误的样本权值和

[GBDT](https://blog.csdn.net/zpalyq110/article/details/79527653)
 - 基本思想：对上一轮的残差进行拟合
 - GBDT中的树是回归树
 - 提升树是迭代多棵回归树来共同决策。当采用平方误差损失函数时，每一棵回归树学习的是之前所有树的结论和残差，拟合得到一个当前的残差回归树（提升树即是整个迭代过程生成的回归树的累加）
 - 梯度提升决策树(GBDT)
 - [简书](https://www.jianshu.com/p/005a4e6ac775)

[xgb](https://blog.csdn.net/a819825294/article/details/51206410)
 - 损失函数加入正则，并进行泰勒展开用到了二阶导（优点：）
 - 分裂点选取：贪心法、近似直方图算法

[GBDT和xgb的区别](https://www.zhihu.com/question/41354392/answer/98658997)
 - GBDT的基分类器是CART，xgb还支持线性分类器
 - xgb损失函数引入二阶泰勒展开和正则项
 - 列抽样
 - 对缺失值的处理。对于特征的值有缺失的样本，xgboost可以自动学习出它的分裂方向
 - 支持并行，特征并行
 - 分裂时用贪心法和近似直方图算法

[lgb优化](https://zhuanlan.zhihu.com/p/25308051)
 - 带深度限制的Leaf-wise的叶子生长策略
 - 直方图算法(做差加速) 和 双边梯度
 - 直接支持类别特征
 - Cache命中率优化


## 7. 类别不平衡问题

1. 欠采样 

代表算法：EasyEnsemble<br>
 - 首先通过从多数类中独立随机抽取出若干子集。
 - 将每个子集与少数类数据联合起来训练生成多个基分类器。
 - 最终将这些基分类器组合形成一个集成学习系统。

2. 过采样

代表算法：<br>

 - Bootstrap少数样本
 - SMOTE(Synthetic Minority Over-sampling Technique)
 - Borderline(SMOTE的一种提升方法)

[过采样参考网址](https://blog.csdn.net/a358463121/article/details/52304670)

3. 阈值移动

基于原始训练集进行学习，但在用训练好的分类器进行预测时，
将再缩放的公式y_new/(1-y_new)=y/(1-y)*(m+/m-) 嵌入到决策过程中，称为“阈值移动”。

## 8. 偏差和方差

泛化误差 = 偏差 + 方差 + 噪声

偏差：偏差度量了学习算法的期望预测与真实结果偏离程度，即刻画了学习算法本身的拟合能力<br>
方差：方差度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响，或者说学习算法的稳定性

**8.1 过拟合即高方差，如何处理**

1. 增加样本数<br>
2. 降低模型的复杂度(CV)。比如决策树模型中降低树深度、进行剪枝等<br>
3. 加正则，使用正则化能够给模型的参数进行一定的约束，避免过拟合(损失函数上)<br>


### 9. 生成模型和判别模型

判别模型总体特征：
 - 对P(Y|X)建模
 - 对所有样本只构建一个模型，确定总体判别边界
 - 优点：对数据量没生成式这么严格，训练速度快，小数据量下准确率也好些

生成模型总体特征：
 - 对P(X, Y)建模，即P(X, Y) = P(X|Y)P(Y)然后预测P(Y|X) = P(X, Y) / P(X)
 - 通过学习来的联合分布P(X, Y)再结合新样本X，通过条件概率求出Y，没有判别边界
 - 优点：所包含的信息全，可以处理隐变量

---

面试总结参考网址：

[牛客网coding](https://www.nowcoder.com/ta/coding-interviews?page=1)

[lintcode](https://www.lintcode.com/problem/)

[牛客网sql](http://www.nowcoder.com/ta/sql)

[机器学习中你不可不知的几个算法常识](https://mp.weixin.qq.com/s/Fh-eQm41DI3rkKjEgC1Yig)

[机器学习、数据挖掘、数据分析岗面试总结](https://blog.csdn.net/Datawhale/article/details/81212235?from=singlemessage&isappinstalled=0)<br>

[机器学习项目流程](https://www.cnblogs.com/wxquare/p/5484690.html)

[Data Science Question Answer](https://github.com/ShuaiW/data-science-question-answer)

[别人的简历（参考）](https://cyc2018.github.io/page.html#next)
