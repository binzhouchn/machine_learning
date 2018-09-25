# 异常检测

[**Loan Default Prediction - Imperial College London**](https://www.kaggle.com/c/loan-default-prediction)

 - 加载数据操作有点过时
 - 特征衍生，主要以组合特征为主（加、减、乘、除等）与label算皮尔逊相关
 - 把test第16列小于1的行index拿出来，然后把101列所对应的这些index进行log1-p转换、训练的101列也进行log1-p转换；16列大于等于1的预测都为0（看ind_tmp）
 - 先用gbc分类0和1，然后对test预测概率大于0.55的值，进行回归预测，回归训练数据用train_y大于0的行训练；
 最后多模型融合，svr、gbr、GaussianProcess