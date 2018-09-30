# 异常检测

## 比赛类

[**1. Loan Default Prediction - Imperial College London**](https://www.kaggle.com/c/loan-default-prediction)

 - 加载数据操作有点过时
 - 特征衍生，主要以组合特征为主（加、减、乘、除等）与label算皮尔逊相关
 - 把test第16列小于1的行index拿出来，然后把101列所对应的这些index进行log1-p转换、训练的101列也进行log1-p转换；16列大于等于1的预测都为0（看ind_tmp）
 - 先用gbc分类0和1，然后对test预测概率大于0.55的值，进行回归预测，回归训练数据用train_y大于0的行训练；
 最后多模型融合，svr、gbr、GaussianProcess
 
 [代码](code/loan_default_prediction/README.md)
 

[**2. 西南财经大学“新网银行杯”数据科学竞赛**](http://www.dcjingsai.com/common/cmpt/%E8%A5%BF%E5%8D%97%E8%B4%A2%E7%BB%8F%E5%A4%A7%E5%AD%A6%E2%80%9C%E6%96%B0%E7%BD%91%E9%93%B6%E8%A1%8C%E6%9D%AF%E2%80%9D%E6%95%B0%E6%8D%AE%E7%A7%91%E5%AD%A6%E7%AB%9E%E8%B5%9B_%E7%BB%93%E6%9E%9C%E6%8F%90%E4%BA%A4.html)

 - 数据集先合并
 - 先可以不进行空缺值去除(0.95)和single_unique，或collinear
 - 先每一行的空缺值进行统计（连续型，类别型分类统计NA，及加总的NA三个特征）
 - 数值型空缺值用均值填充（中位数也可以，对异常值不敏感）
 - 先跑一个base lgb，然后根据重要排序选择15%-20%重要特征
 - 特征衍生...
 - 根据重要的特征进行特征衍生(poly特征)
 - 根据原始特征进行组合特征(pls,sub,mul,div,sub_mul等)
 

---
 
## 项目类

