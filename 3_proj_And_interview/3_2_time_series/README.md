# 时间序列

## 比赛类 statistical method


[待补充] 以后看到好的时间序列比赛及kernel再补充


## 比赛类 machine learning method

[**1. Corporación Favorita Grocery Sales Forecasting**](https://www.kaggle.com/c/favorita-grocery-sales-forecasting)

#### 1.1 lgbm-one-step-ahead.py

介绍：代码中包含<br>
 - set_index：对数据进行店铺到商品的三级目录操作<br>
 - stack/unstack：把数据进行平铺<br>
 - pd.date_range/date/timedelta：对间隔时间列进行提取get_timespan
 并进行相关的统计特征操作<br>
 - 训练集，验证集，测试集<br>
训练集X是2017/5/31号之前的一些时间切片特征，y是2017/5/31-6/15的销量作为label；
训练集可以增加一些，通过每隔7天操作，然后concat(axis=0)<br>
验证集X是2017/7/26的时间切片特征，y是7/26-8/10的销量
测试集就是需要预测的是8/16-8/31的销量，用的X是8/15之前的一些时间切片特征<br>
 - 预测的时候，X不变，每次预测lgb.train完了以后预测一天的销量，总共预测16次
 最后进行stack操作16列变成一列，提交结果
 
 [代码](code/lgbm-one-step-ahead.py)
 
#### 1.2 lstm-starter

 - 基本思路和上一个一样，就是模型换成了lstm
 
 [keras版本代码](code/lstm-starter-keras-version.py)
 
 [pytorch版本代码-byself](code/lstm_starter_pytorch_version)
 - 重写损失函数
 - 加入cv
 
---

## 项目类

待补充..


---

参考资料

待补充..