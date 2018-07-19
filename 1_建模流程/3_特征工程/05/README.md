## 5. 降维

- 当特征选择完成后，可以直接训练模型了，但是可能由于特征矩阵过大，导致计算量大，训练时间长的问题，因此降低特征矩阵维度也是必不可少的。
- 常见的降维方法除了以上提到的基于L1惩罚项的模型以外，另外还有主成分分析法（PCA）和线性判别分析（LDA），线性判别分析本身也是一个分类模型。<br>
- PCA和LDA有很多的相似点，其本质是要将原始的样本映射到维度更低的样本空间中，但是PCA和LDA的映射目标不一样：`PCA是为了让映射后的样本具有最大的发散性；而LDA是为了让映射后的样本有最好的分类性能。`所以说PCA是一种无监督的降维方法，而LDA是一种有监督的降维方法。

--- 

**PCA，主成分分析，计算协方差矩阵**

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=15) # n_components : 设置留下来几列
newdata = pca.fit_transform(xdata)

#pca.explained_variance_            # 可解释方差
#pca.explained_variance_ratio_      # 可解释方差百分比
```

\*注意：PCA 前先将数据进行标准化，用z-score即可!!!

**TruncatedSVD，截断SVD**

TruncatedSVD 原来N列 可以选择指定保留k列 , 降维<br>
```python
from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=15, algorithm='randomized')
#n_components：int  , 输出数据的期望维度。
```
