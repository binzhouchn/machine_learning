[<h1 align = "center">:alien: 模型集成 :alien:</h1>][0]

---

### 1. Averaging 和 Voting
- 平均法 对于目标变量为连续值的任务，使用平均
- 投票法
    - 硬投票：每个模型输出它自认为最可能的类别，投票模型从其中选出投票模型数量最多的类别，作为最终分类。
    - 软投票：每个模型输出一个所有类别的概率矢量(1 * n_classes)，投票模型取其加权平均，得到一个最终的概率矢量。

### 2. Stacking

![stacking](stacking.png)

上图展示了使用5-Fold进行一次Stacking的过程<br>
主要步骤是，比如数据是200个特征，样本数是10万个，base model经过5折cv以后得到10万个预测值（即生成一个新特征）<br>
多个基模型就有了多个特征，最后再跑一个模型


---
## 2. 多样性
- 误差——分歧分解
- 多样性度量
- 多样性增强
    - 数据样本扰动
    - 输入属性扰动
    - 算法参数扰动
    - 输出表示扰动
        - 翻转法(Flipping Output)：随机改变一些训练样本标记
        - 输出调制法(Output Smearing)：分类输出转化为回归输出
        - OVO/ECOC


---
[0]: http://www.cnblogs.com/jasonfreak/p/5657196.html
