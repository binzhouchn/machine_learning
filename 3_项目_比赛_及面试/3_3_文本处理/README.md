# 文本处理

## 比赛类

[**1. 搜狗用户画像**](https://www.datafountain.cn/competitions/239/details)

## 1.1 tfidf知识点

这里默认的tokenizer就是用空格进行分割单词的
```python
from sklearn.feature_extraction.text import TfidfVectorizer
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

[**2. 汽车行业用户观点主题及情感识别**](https://www.datafountain.cn/competitions/310/details)

[初步代码,跑一个lstm流程](0_汽车行业用户观点主题及情感识别_lstm_naive.ipynb)<br>
[初步代码,跑一个cnn流程](1_汽车行业用户观点主题及情感识别_cnn_naive.ipynb)<br>
[胶囊网络Capsule keras版 来自312shan](2_汽车行业用户观点主题及情感识别_capsule_keras.ipynb)<br>
[胶囊网络Capsule pytorch版(通过keras版复现)](https://github.com/binzhouchn/capsule-pytorch)


[别人的代码baseline 62+](https://github.com/312shan/Subject-and-Sentiment-Analysis)

---

## 项目类


[**1. 智能客服**](智能客服流程.md)


[**2. 催收机器人**](催收机器人.md)


---

参考资料：

[停用词库](https://github.com/goto456/stopwords)
