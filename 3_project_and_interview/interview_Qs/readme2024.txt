宏平均和微平均是用来干什么的？是如何计算的？他俩的主要区别？
https://zhuanlan.zhihu.com/p/646328218
https://blog.csdn.net/qq_34126716/article/details/108704124
知识蒸馏为什么有效. 详细介绍一下知识蒸馏？
https://blog.csdn.net/qq_41980734/article/details/128837062
Transformer为何能够有效地处理长距离依赖问题?与传统RNN和LSTM相比有哪些优势?
答：自注意力机制；最大优势：并行计算，即不同位置的信息可以同时进行处理，加快了训练速度
多头注意力的作用是什么?
答：这种设计使模型能够更好地理解语言的多种复杂关系
在Transformer模型中，位置编码(Position Encoding)的作用是什么？
答：使用位置编码来保留有关句子中单词顺序的信息
Bert中有哪些地方用到了mask？
答：padding mask + attention mask
https://www.zhihu.com/question/404452350/answer/1323386941
预训练阶段的mask有什么用？
答：mask这种方式鼓励模型学习上下文信息，理解元素之间的关系，从而提升模型的表征能力。
https://blog.csdn.net/qq_40427481/article/details/132580591
Bert中的transformer和原生的transformer有什么区别？
https://answer.baidu.com/answer/land?params=20zvC4xT5m9zuRbwHusuJ%2F6NMXW70f6QPaKq65kSMSG26GKA8excIHdrl0KBzSt8mT%2BqOpwHYQWMGDVaW4fhg9H24XV0gFDuh7VcxmhHX0yW5YIoSvN8lW1Ql3%2B6HbzRb0Azdxz5EMROnFuoddq91UZumKraLw%2FuSlMQ2hiJjIxCuTWvDq6e7B3fncXvDPTz3XsxS5Tzs%2By4e96f7jhZ363Irhg3gBKcV5AjDkJosNNyRJyhtGyW0KV3QleNCkdtXzJ9LDaatuUsAAB7unVWsg%3D%3D&from=dqa&lid=de1c3e5f009f1e5c&word=Bert%E4%B8%AD%E7%9A%84transformer%E5%92%8C%E5%8E%9F%E7%94%9F%E7%9A%84transformer%E6%9C%89%E4%BB%80%E4%B9%88%E5%8C%BA%E5%88%AB%3F
强化学习适合在哪些场景使用？
答：理论上大部分场景都可以使用，目前用的比较多的应该就是决策类的比如围棋、游戏、自动驾驶等领域吧
智力题：如何用rand6实现rand10

代码题:
最小覆盖子串：给你一个字符串 s 、一个字符串 t 。返回 s 中涵盖 t 所有字符的最小子串。如果 s 中不存在涵盖 t 所有字符的子串，则返回空字符串 ""
答：https://www.lintcode.com/problem/32/
----------
layer normalization 的好处是？和 batch normalization 的区别？你有没有看过有的 transformer 也用bn？
BERT 的主要改进，包括结构的改进. 预训练方式的改进，都说一下？
Reformer中. LSH（局部敏感哈希）是如何实现的？
CRF 和 HMM 的区别. 哪个假设更强？他们的解码问题用什么算法？
lstm 参数量计算. 给出 emb_size 和 hidden_size. 求lstm参数量
简单实现一个layer normalization类. 只需要实现__init__和forward就行
简述GPT和BERT的区别
为什么现在的大模型大多是decoder-only的架构？
如何减轻LLM中的幻觉现象？
GPT-3拥有的1750亿参数，是怎么算出来的？
参数高效的微调（PEFT）有哪些方法？
目前主流的中文向量模型有哪些？
答：word2vec,glove,text2vec,m3e,bge-m3
请介绍一下微软的ZeRO优化器
答：https://www.bilibili.com/video/BV1fb421t7KN/?spm_id_from=333.788&vd_source=71b0ee5f5f86cfe144c869f87c36fe4f（推荐）
https://zhuanlan.zhihu.com/p/690598696

代码题:
反转字符串和反转字符串II
答：leetcode
无重复字符的最长字串
答：https://www.lintcode.com/problem/384/


附录：
[大模型算法ms - 基础篇](https://blog.csdn.net/sinat_37574187/article/details/135578853)


代码刷题请参考自建仓库[algo](https://github.com/binzhouchn/algor)<br>
https://www.lintcode.com/problem
https://leetcode.cn