# 金融风控

## 比赛类

按照时间顺序从最新开始，可以重点参考下第一个!

[1. 第二届YIZHIFU杯大数据建模大赛-信用风险用户识别](https://www.dcjingsai.com/v2/cmptDetail.html?id=410)

 - EDA
 - 特征工程
 - lgb建模
 
 [**代码**](code/YIZHIFU)
 
[2. IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection/overview)

 参考下这个风控比赛的baseline包括eda、建模等

[3. Loan Default Prediction - Imperial College London](https://www.kaggle.com/c/loan-default-prediction)

 - 加载数据操作有点过时
 - 特征衍生，主要以组合特征为主（加、减、乘、除等）与label算皮尔逊相关
 - 把test第16列小于1的行index拿出来，然后把101列所对应的这些index进行log1-p转换、训练的101列也进行log1-p转换；16列大于等于1的预测都为0（看ind_tmp）
 - 先用gbc分类0和1，然后对test预测概率大于0.55的值，进行回归预测，回归训练数据用train_y大于0的行训练；
 最后多模型融合，svr、gbr、GaussianProcess
 
 [**代码**](code/loan_default_prediction/README.md)
 

[4. 西南财经大学“新网银行杯”数据科学竞赛](http://www.dcjingsai.com/common/cmpt/%E8%A5%BF%E5%8D%97%E8%B4%A2%E7%BB%8F%E5%A4%A7%E5%AD%A6%E2%80%9C%E6%96%B0%E7%BD%91%E9%93%B6%E8%A1%8C%E6%9D%AF%E2%80%9D%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%AB%9E%E8%B5%9B_%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4.html)

 - 数据集先合并
 - 粗筛：空缺值去除(0.95)和single_unique，或collinear
 - 先每一行的空缺值进行统计（连续型，类别型分类统计NA，及加总的NA三个特征）
 - 相同空缺率的数值型特征进行横向加总
 - 数值型空缺值用均值填充（中位数也可以，对异常值不敏感）
 - 先跑一个base lgb，然后根据重要排序选择15%-20%重要特征
 - 特征衍生START.............................................
 - 根据重要的数值特征进行特征衍生(poly特征)
 - 根据原始特征进行组合特征(pls,sub,mul,div,sub_mul等) loan_default_pred代码中有
 - 原始数据跑孤立森林，衍生一列-1或1的特征
 - NA的排序特征(行)（optional可能有用）
 - 所有数据处理完后可以跑一下gbdt特征
 - FFM分类概率（好像比LR要好一点）
 - 特征衍生END...........................................
 - 跑模型START...........................................
 - 可以根据相关性再细筛一下
 - 跑lgb
 - 针对类别型变量再onehot以后跑下LR，ADABOOST, EXTRATREE, RF
 - 然后进行结果简单融合
 - 跑模型END............................................
 
[**代码**](code/西南财经大学_新网银行杯)

[5. 融360天机 智能金融算法挑战赛](http://openresearch.rong360.com/#/question)

 - [第一题 拒绝推断 top2方案](https://zhuanlan.zhihu.com/p/46090290)
 - [第二题 特征挖掘 1st_解决方案](https://github.com/xSupervisedLearning/Rong360_feature_mining_1st_solution)
 - [第二题 特征挖掘 2nd_解决方案 有网盘数据](https://github.com/Questions1/Rong360_2nd)
 - [第三题 多金融场景下的模型训练 top1解决方案](https://github.com/shuiliwanwu/Rong360-Model-training-in-multiple-financial-scenarios)

---
 
## 项目类

[**1. 评分卡算法建模流程20180901**](评分卡算法建模流程.md)

[**2. 评分卡算法建模流程20181001(补充完整流程)**](code/scorecardpy-master/README.md)

---

参考资料
 
[金融风险控制基础常识——巴塞尔协议+信用评分卡Fico信用分](https://blog.csdn.net/sinat_26917383/article/details/51720662)<br>
[风控分类模型种类（决策、排序）比较与模型评估体系（ROC/gini/KS/lift）](https://blog.csdn.net/sinat_26917383/article/details/51725102)<br>
[信用风险模型（申请评分、行为评分）与数据准备（违约期限、WOE转化）](https://blog.csdn.net/sinat_26917383/article/details/51721107)<br>
[ABC卡](https://blog.csdn.net/Eason_oracle/article/details/78602914)<br>

