# 异常检测

[**Loan Default Prediction - Imperial College London**](https://www.kaggle.com/c/loan-default-prediction)

 - 加载数据有点过时
 - 特征衍生，主要以组合特征为主（加、减、乘、除等）与label算皮尔逊相关
 - 
 - 先用gbc分类0和1，然后对预测概率大于0.55的值，进行回归预测
 最后多模型融合，svr、gbr、GaussianProcess