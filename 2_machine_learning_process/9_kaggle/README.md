## 股票市场波动率预测

https://www.kaggle.com/c/optiver-realized-volatility-prediction

[my work: lb 0.19561](https://www.kaggle.com/binzhouchn/latest-code9-final-1)<br>
[CatBoost | LGBM x 2 | Hyperparameter Tuning](https://www.kaggle.com/oldwine357/catboost-lgbm-x-2-hyperparameter-tuning/notebook#LGBM-Tuning)<br>
[Deanonimising time from event](https://www.kaggle.com/lucasmorin/deanonimising-time-from-event)<br>
[baseline-denoiseautoencoder-1dcnn: lb 0.20136](https://www.kaggle.com/binzhouchn/baseline-denoiseautoencoder-1dcnn)<br>
[Optiver 2LSTM+Attention is incoming: lb 0.22526](https://www.kaggle.com/vv0x0x/optiver-2lstm-attention-is-incoming/comments)<br>
[LGBM Hyperparameter Search with Genetic Algorithms](https://www.kaggle.com/ollibolli/lgbm-hyperparameter-search-with-genetic-algorithms)<br>


[stacking简单参考](https://zhuanlan.zhihu.com/p/69714954)<br>


## LLM - Detect AI Generated Text

经典代码片段<br>
```python
#使用传统机器学习方法，集成方法
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier 
from sklearn.naive_bayes import MultinomialNB  
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
# from sklearn.feature_extraction.text import TfidfVectorizer

def get_model():
    # Multinomial Naive Bayes classifier with specified alpha
    clf = MultinomialNB(alpha=0.0225)

    # Stochastic Gradient Descent (SGD) classifier with custom settings
    sgd_model = SGDClassifier(max_iter=9000, tol=1e-4, loss="modified_huber", random_state=6743)

    # LightGBM classifier with hyperparameters defined in dictionary p6
    p6 = {'n_iter': 3000, 'verbose': -1, 'objective': 'cross_entropy', 'metric': 'auc',
          'learning_rate': 0.00581909898961407, 'colsample_bytree': 0.78, 'colsample_bynode': 0.8}
    p6["random_state"] = 6743
    lgb = LGBMClassifier(**p6)

    # CatBoost classifier with specified iterations, learning rate, and subsample
    cat = CatBoostClassifier(iterations=3000, verbose=0, random_seed=6543,
                             learning_rate=0.005599066836106983, subsample=0.35,
                             allow_const_label=True, loss_function='CrossEntropy')

    # Weights for the Voting Classifier, specifying the importance of each base classifier
    weights = [0.1, 0.31, 0.28, 0.67]

    # Create a Voting Classifier with the specified base classifiers and weights
    ensemble = VotingClassifier(estimators=[('mnb', clf),
                                            ('sgd', sgd_model),
                                            ('lgb', lgb),
                                            ('cat', cat)
                                            ],
                                weights=weights, voting='soft', n_jobs=-1)
    return ensemble
```