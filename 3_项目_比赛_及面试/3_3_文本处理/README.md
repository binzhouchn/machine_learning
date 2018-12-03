# 文本处理nlp

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

---

## 2. 比赛类

[**2.1 搜狗用户画像**](https://www.datafountain.cn/competitions/239/details)

## 2.1.1 tfidf知识点

这里默认的tokenizer就是用空格进行分割单词的
```python
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.feature_extraction.text import CountVectorizer # CountVectorizer和tfidf类似，用词频表征
corpus = ['this is test it','this unbeliveble haha']
vectorizer = TfidfVectorizer()
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

