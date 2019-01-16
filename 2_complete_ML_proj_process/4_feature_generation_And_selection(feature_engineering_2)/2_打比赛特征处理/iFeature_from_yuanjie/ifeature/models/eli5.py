# -*- coding: utf-8 -*-
"""
__title__ = 'eli5'
__author__ = 'JieYuan'
__mtime__ = '2018/8/21'
"""
from eli5.lime import TextExplainer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


X = ["The dimension of the input documents is reduced to 100, and then a kernel SVM is used to classify the documents.",
     "This is what the pipeline returns for a document - it is pretty sure the first message in test data belongs to sci.med:"]

y = [0, 1]

piplie = make_pipeline(TfidfVectorizer(), LogisticRegression())

te = TextExplainer(random_state=42)
te.fit(X[0], piplie.predict_proba)
te.show_prediction()
te.show_weights()


eli5.show_prediction
