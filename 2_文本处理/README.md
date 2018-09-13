# 文本处理

[**搜狗用户画像**]

### 1. tfidf

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

