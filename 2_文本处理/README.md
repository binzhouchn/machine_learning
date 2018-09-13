# 文本处理

[**搜狗用户画像**]

### 1. tfidf

这里默认的tokenizer就是用空格进行分割单词的
```python
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = ['this is test it','this unbeliveble haha']
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)
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
            words = [word for word in jieba.cut(query)]
            for gram in [1,2]:
                for i in range(len(words) - gram + 1):
                    tokens += ["_*_".join(words[i:i+gram])]
        if np.random.rand() < 0.00001:
            print(line)
            print('='*20)
            print(tokens)
        self.n += 1
        if self.n%10000==0:
            print(self.n,end=' ')
        return tokens 
```

