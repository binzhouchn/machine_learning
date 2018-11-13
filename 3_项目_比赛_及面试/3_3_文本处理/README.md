# 文本处理

## Basics

### 1. gensim加载自己的训练的词向量

```python
#文件格式，存成txt文件
'''
8383 100
我 -0.027192615 0.3832912 0.29597545 -0.15442707 0.13949695 0.37888202 -0.070740506 0.16849327 -0.00089764595 0.022406599 0.08953266 -0.20218499 -0.21548781 0.1358894 
'''
glove_model= gensim.models.KeyedVectors.load_word2vec_format('./ft_wv.txt')
```


### 2. 各种距离计算及文本相似度算法

[各种距离计算及文本相似度算法](各种距离计算及文本相似度算法.py)


---

## 比赛类

[**搜狗用户画像**](https://www.datafountain.cn/competitions/239/details)

## 1.1 tfidf知识点

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
        if self.n%10000==0:
            print(self.n,end=' ') # 每10000条打印一下
        return tokens 
```

[**汽车行业用户观点主题及情感识别**](https://www.datafountain.cn/competitions/310/details)

[初步代码,跑一个lstm流程](pytorch_code/汽车行业用户观点主题及情感识别_lstm_naive.ipynb)<br>
[初步代码,跑一个cnn流程](pytorch_code/汽车行业用户观点主题及情感识别_cnn_naive.ipynb)<br>
[胶囊网络Capsule keras版 来自312shan](pytorch_code/2_汽车行业用户观点主题及情感识别_capsule_keras.ipynb)<br>
[胶囊网络Capsule pytorch版(通过keras版复现)](https://github.com/binzhouchn/capsule-pytorch)


[别人的代码baseline 62+](https://github.com/312shan/Subject-and-Sentiment-Analysis)


---

## 项目类


[**1. 智能客服**](智能客服流程.md)


[**2. 催收机器人**](催收机器人.md)


---

参考资料：

[停用词库](https://github.com/goto456/stopwords)

[腾讯AI Lab开源大规模高质量中文词向量数据](https://cloud.tencent.com/developer/article/1356164)

[各种距离](https://blog.csdn.net/shiwei408/article/details/7602324)

