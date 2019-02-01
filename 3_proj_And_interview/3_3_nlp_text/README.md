# 文本处理nlp

NLP四大类任务<br>
 - 序列标注：分词/POS Tag/NER
 - 分类任务：文本分类/情感分析
 - 句子关系判断：Entailment/自然语言推理（它的特点是给定两个句子，模型判断出两个句子是否具备某种语义关系）
 - 生成式任务：机器翻译/闲聊/文本摘要/写诗造句/看图说话


## 1. Basics

### 1.1 gensim加载自己的训练的词向量

```python
#文件格式，存成txt文件
'''
8383 100
我 -0.027192615 0.3832912 0.29597545 -0.15442707 0.13949695 0.37888202 -0.070740506 0.16849327 -0.00089764595 0.022406599 0.08953266 -0.20218499 -0.21548781 0.1358894 
'''
glove_model= gensim.models.KeyedVectors.load_word2vec_format('./ft_wv.txt')
```

### 1.2 各种距离计算及文本相似度算法

 - 欧式距离
 - 曼哈顿距离
 - 切比雪夫距离
 - 闵可夫斯基距离
 - 马氏距离
 - 编辑距离（edit distance）
 - 余弦相似性（cosine similarity）
 - WMD距离（word mover’s distance）
 - 杰卡顿距离（Jaccard distance）

[各种距离计算及文本相似度算法](各种距离计算及文本相似度算法.py)

**余弦相似度**<br>
大数据量时的余弦计算<br>
1.分母向量的长度不需要重复计算<br>
2.分子，向量内积的时候，只需考虑向量中的非零元素，下降100倍<br>
3.删除虚词，的、是、和，及一些连词、副词和介词

**simhash**<br>
[简单易懂讲解simhash算法 hash 哈希](https://blog.csdn.net/le_le_name/article/details/51615931)



### 1.3 N-Gram:简单的马尔科夫链

bigram:一个词的出现仅依赖于它前面出现的一个词 P(w1,w2...wm) = 连乘P(wi|wi-1) 假设有一个很大的语料库，我们统计下面一些词出现的量，其中I出现了2533次，
再给出基于bigram模型进行计数的结果，其中第一行，第二列表示给定前一个词是"I"时，当前词为"want"的情况一共出现了827次，所以P(want|I)=827/2533=0.33

ngram的应用：搜索，输入法联想，文本自动生成<br>
[自然语言处理中的N-Gram模型详解](https://blog.csdn.net/baimafujinji/article/details/51281816)

**生成ngram**<br>
```python
def generate_ngram(input_list, n):
    result = []
    for i in range(1, n+1):
        result.extend(zip(*[input_list[j:] for j in range(i)]))
    return result
    
generate_ngram([1,2,3,4,5], 3)
# [(1,), (2,), (3,), (4,), (1, 2), (2, 3), (3, 4), (1, 2, 3), (2, 3, 4)]
```
或者用nltk包
```python
from nltk.util import ngrams
list(ngrams([1,2,3,4], 3))
# [(1, 2, 3), (2, 3, 4)]
```

### 隐马尔可夫HMM和Viterbi算法

什么是HMM，隐含马尔可夫模型的三个基本问题（我的理解）：<br>
(1) 给定一个模型，如何计算某个特定的输出序列的概率？<br> 
答：遍历算法、向前算法、向后算法 [楼二 henry](https://www.zhihu.com/question/20962240?sort=created)<br>
(2) 给定一个模型和某个特定的输出序列，找到最可能产生这个输出的(隐)状态序列？<br>
答：我们根据转移概率P(St|St-1)和发射概率P(ot|st)
列举所有可能的隐状态序列，选最大概率的序列；现实中这个情况太多所以
通常用维特比算法解决（维特比算法就是求解HMM上的最短路径（-log(prob)，也即是最大概率）的算法）<br>
[小白给小白详解维特比算法](https://blog.csdn.net/athemeroy/article/details/79339546)<br>
[维特比算法python代码](http://www.hankcs.com/nlp/hmm-and-segmentation-tagging-named-entity-recognition.html)<br>
(3) 给定足够量的观测数据，如何估计隐含马尔可夫模型的参数？<br>
训练隐含马尔可夫模型更实用的方式是仅仅通过大量观测到的信号O1，O2，O3，….就能推算模型参数的P(St|St-1)和P(Ot|St)的方法（无监督训练算法），
其中主要使用[鲍姆-韦尔奇算法](https://www.cnblogs.com/pinard/p/6972299.html)

原理：三个骰子例子【楼一】 天气模型【楼二】<br>
[如何用简单易懂的例子解释隐马尔可夫模型](https://www.zhihu.com/question/20962240?sort=created)

NLP应用：<br>
具体到分词系统，可以将天气当成“标签”，活动当成“字或词”。那么，几个NLP的问题就可以转化为：<br>
 - 词性标注：给定一个词的序列（也就是句子），找出最可能的词性序列（标签是词性）。如ansj分词和ICTCLAS分词等。<br>
 - 分词：给定一个字的序列，找出最可能的标签序列（断句符号：[词尾]或[非词尾]构成的序列）。结巴分词目前就是利用BMES标签来分词的，B（开头）,M（中间),E(结尾),S(独立成词）
 - 命名实体识别：给定一个词的序列，找出最可能的标签序列（内外符号：[内]表示词属于命名实体，[外]表示不属于）。如ICTCLAS实现的人名识别、翻译人名识别、地名识别都是用同一个Tagger实现的。

[HMM与分词、词性标注、命名实体识别](http://www.hankcs.com/nlp/hmm-and-segmentation-tagging-named-entity-recognition.html)

### 1.4 中文新词发现(和分词)

[python3实现互信息和左右熵的新词发现](https://blog.csdn.net/qq_34695147/article/details/80464877)

互信息（凝固度）<br>
左右熵（自由度）<br>
新词IDF

**分词：**<br>
加载搜狗词典sogou_words.txt<br>
[加载腾讯词典tencent_words.txt](code/文本分类_tfidf+bert+Tencent词向量/README.md)<br>
[利用互信息和左右信息熵的中文分词新词发现github](https://github.com/zhanzecheng/Chinese_segment_augment)

### 1.5 关键词提取主要算法

 - tfidf
 - topic-model(LDA)
 - textrank关键词提取
 
[关键词提取详细笔记](https://github.com/binzhouchn/ai_notes/blob/master/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/%E5%85%AC%E5%85%B1%E6%A8%A1%E5%9D%97/keyword_extraction.md)

### 1.6 word2vec

 - CBOW模型
 - Skip-gram模型
 - Hierarchical Softmax 与 Negative Sampling
 - FastText
 - GolVe

GloVe 与 Word2Vec 的区别:

 - Word2Vec 本质上是一个神经网络；Glove 也利用了反向传播来更新词向量，但是结构要更简单，所以 GloVe 的速度更快
 - Glove 认为 Word2Vec 对高频词的处理还不够，导致速度慢；GloVe 认为共现矩阵可以解决这个问题
 - 从效果上看，虽然 GloVe 的训练速度更快，但是词向量的性能在通用性上要弱一些

FastText 是从 Word2Vec 的 CBOW 模型演化而来的，不同点：

 - CBOW 的输入是中心词两侧skip_window内的上下文词；FastText 除了上下文词外，还包括这些词的字符级 N-gram 特征
 
[word2vec详细笔记](https://github.com/binzhouchn/ai_notes/blob/master/%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/%E5%85%AC%E5%85%B1%E6%A8%A1%E5%9D%97/word2vec.md)

### 1.7 Transformer

[[整理]聊聊 Transformer](https://zhuanlan.zhihu.com/p/47812375)

---

## 2. 比赛类

[**2.1 搜狗用户画像**](https://www.datafountain.cn/competitions/239/details)

## 2.1.1 tfidf知识点

这里默认的tokenizer就是用空格进行分割单词的
```python
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer和tfidf类似，用词频表征
corpus = ['this is test it','this unbeliveble haha']
vectorizer = TfidfVectorizer() # 默认每句话中的词用空格分开，也可以自定义tokenizer
tfidf = vectorizer.fit_transform(corpus)

tfidf.todense() # 把稀疏矩阵转成稠密矩阵，每个文档就是一个向量
# 查看单词及单词对应的索引
vectorizer.get_feature_names()
vectorizer.vocabulary_
```

可以改写成tokenizer比如用\t进行分割，重写__call__函数即可
```python
class Tokenizer:
    def __init__(self):
        pass
    def __call__(self, line):
        tokens = line.split('\t')
        return tokens

from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['this\tis\ttest\tit','this\tunbeliveble\thaha']
vectorizer = TfidfVectorizer(tokenizer=Tokenizer())
tfidf = vectorizer.fit_transform(corpus)
```

也可以给字符串除了分词还可以增加一些其他的token
```python
class Tokenizer():
    def __init__(self):
        self.n = 0
    def __call__(self, line):
        tokens = []
        for query in line.split('\t'):
            words = jieba.lcut(query)
            for gram in [1,2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i+gram])]
        self.n += 1
        if self.n % 10000 == 0:
            print(self.n, end=' ') # 每10000条打印一下
        return tokens 
```

[**2.2 汽车行业用户观点主题及情感识别**](https://www.datafountain.cn/competitions/310/details)

[初步代码,跑一个lstm流程](pytorch_code/汽车行业用户观点主题及情感识别_lstm_naive.ipynb)<br>
[初步代码,跑一个cnn流程](pytorch_code/汽车行业用户观点主题及情感识别_cnn_naive.ipynb)<br>
[胶囊网络Capsule keras版 来自312shan](pytorch_code/2_汽车行业用户观点主题及情感识别_capsule_keras.ipynb)<br>
[胶囊网络Capsule pytorch版(通过keras版复现)](https://github.com/binzhouchn/capsule-pytorch)


[别人的代码baseline 62+](https://github.com/312shan/Subject-and-Sentiment-Analysis)


---

## 3. 项目类

[**3.1 智能客服**](智能客服流程.md)

[**3.2 催收机器人**](催收机器人.md)


---

参考资料：

[停用词库](https://github.com/goto456/stopwords)

[腾讯AI Lab开源大规模高质量中文词向量数据](https://cloud.tencent.com/developer/article/1356164)

[各种距离](https://blog.csdn.net/shiwei408/article/details/7602324)

