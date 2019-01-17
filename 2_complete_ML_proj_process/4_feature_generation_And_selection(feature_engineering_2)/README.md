# 特征衍生和筛选

【有agg的情况 聚合】（多列，id重复）

数值型特征可以有统计特征，排序特征和多项式特征<br>

数值型特征：count sum avg std min max mean max-min等

类别型特征：count nunique category_density等


---

【没有agg的情况 未聚合】（单列，id不重复）

 - 数值列 
    - 缺失值特征
    - 异常值特征：3sigma/箱型图，孤立森林(isolated forest)
    - 分箱
    - 多项式特征
 - 类别型特征 
    - 计数和排序特征
    - （交叉特征）：结合类别型特征对数值型进行编码
 - 时间特征
 - 组合特征（强特组合）
 
 