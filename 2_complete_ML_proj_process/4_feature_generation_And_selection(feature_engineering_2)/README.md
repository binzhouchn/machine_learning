# 特征衍生和筛选

【有agg的情况 聚合】（多列，id重复）

数值型特征可以有统计特征，排序特征和多项式特征<br>

统计特征：count sum avg std min max mean max-min等

多项式特征：

类别型特征：count nunique category_density等


---

【没有agg的情况 未聚合】（单列，id不重复）

数值列， 一般只能做缺失值特征，异常值特征，分箱

类别型特征（交叉特征）：结合类别型特征对数值型进行编码

异常特征：3sigma/箱型图，孤立森林(isolated forest)

多项式特征：

时间特征：