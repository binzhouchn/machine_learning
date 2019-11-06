# -*- coding: utf-8 -*-
"""
__title__ = '_model_cv'
__author__ = 'JieYuan'
__mtime__ = '2018/7/20'
"""

from sklearn.model_selection import StratifiedKFold, cross_val_score


def model_cv(clf, X, y, nb_cv=3, seed=42):
    scores = cross_val_score(clf, X, y,
                             scoring='roc_auc',
                             cv=StratifiedKFold(nb_cv, True, seed),
                             n_jobs=10)
    print('\n', scores)
    print(f'{scores.mean()}', f'+/-{scores.std()}', sep='\t')
