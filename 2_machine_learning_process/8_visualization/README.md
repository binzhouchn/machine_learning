<h1 align = "center">:helicopter: 效果可视化 :running:</h1>


# 1. 针对类别型变量画正负样本的直方图

```python
# 离散型变量的分布情况
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("darkgrid",{"font.sans-serif":['simhei', 'Arial']})
plt.figure(figsize=(20, 10))
sns.countplot(x='sex', hue='label',  data=data)
plt.show()
# 连续型变量查看分布
melt = pd.melt(data, id_vars=['label'], value_vars = [f for f in f_num])
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.stripplot, 'label', 'value', jitter=True, palette="muted")
plt.show()
```

# 2. 针对数值型变量画正负样本的差异图

```python
#每一个特征和label之间的相关可视化
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

target_mask = train['target'] == 1
non_target_mask = train['target'] == 0 
statistics_array = []
for col in train.columns[2:]:
    statistic, pvalue = ks_2samp(train.loc[non_target_mask, col], train.loc[target_mask, col])
    statistics_array.append(statistic)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    sns.kdeplot(train.loc[non_target_mask, col], ax=ax, label='Target == 0')
    sns.kdeplot(train.loc[target_mask, col], ax=ax, label='Target == 1')

    ax.set_title('name: {}, statistics: {:.5f}, pvalue: {:5f}'.format(col, statistic, pvalue))
    plt.show()
```


---
https://github.com/reiinakano/scikit-plot
https://github.com/edublancas/sklearn-evaluation
