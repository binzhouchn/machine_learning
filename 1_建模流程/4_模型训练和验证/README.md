<h1 align = "left">:alien: 模型训练和验证 :alien:</h1>

---

[各种算法优缺点1](https://mp.weixin.qq.com/s?__biz=MzA4OTg5NzY3NA==&mid=2649345665&idx=1&sn=000c6e1ceada252162b803404d9a397c&chksm=880e8124bf790832dfc5b10e142425969799639743295078ee1d9524ab21e7ad1b314136d923&mpshare=1&scene=1&srcid=0528p1yaSx6dNlRh0U58XebG#rd)<br>
[各种算法优缺点2](https://mp.weixin.qq.com/s/6hD19wWEex-0s-dweuP5sg)<br>
[各种算法优缺点3](https://blog.csdn.net/u012422446/article/details/53034260)<br>


### 经验参数

![经验参数](经验参数.jpg)

### 模型选择

 - 对于稀疏型特征（如文本特征，One-hot的ID类特征），我们一般使用线性模型，譬如 Linear Regression 或者 Logistic Regression。Random Forest 和 GBDT 等树模型不太适用于稀疏的特征，但可以先对特征进行降维（如PCA，SVD/LSA等），再使用这些特征。稀疏特征直接输入 DNN 会导致网络 weight 较多，不利于优化，也可以考虑先降维，或者对 ID 类特征使用 Embedding 的方式
 
 - 对于稠密型特征，推荐使用 XGBoost 进行建模，简单易用效果好
 
 - 数据中既有稀疏特征，又有稠密特征，可以考虑使用线性模型对稀疏特征进行建模，将其输出与稠密特征一起再输入XGBoost/DNN建模，具体可以参考5_模型集成中Stacking部分
 
### 调参和模型验证

 - 训练集和验证集的划分
 
 - 指定参数空间
 
 - 按照一定的方法进行参数搜索

[具体展开看此链接](https://m.sohu.com/a/139981834_116235)

