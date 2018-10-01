# -*- coding: utf-8 -*-
"""
__title__ = 'SemiSupervision'
__author__ = 'JieYuan'
__mtime__ = '2018/8/17'
"""
import numpy as np

from .baseline import Model


class SemiSupervision(object):
    def __init__(self):
        pass

    def regression(self, X_train, y_train, X_test, learning_rate=0.1, iteration=3, feval=None, seed=None):

        for i in range(iteration + 1):
            if i == 0:
                print("\n初始化模型 ...")
                X, y = X_train, y_train
            else:
                print("\n%2d 迁移学习/半监督 ..." % i)
                X = np.row_stack([X_train, X_test])
                y = np.hstack([y_train, m.model.predict(X_test)])

            m = Model(X, y, learning_rate=learning_rate, metric='rmse', application='regression', feval=feval, seed=seed)
            m.cv(feval, return_model=True)

        return m.model

        # def classifier(self, X_train, y_train, X_test, iteration=3, feval=None):
        #
        #     for i in tqdm(range(iteration + 1), 'Iteration ...'):
        #         if i == 0:
        #             print("\n初始化模型 ...")
        #             X, y = X_train, y_train
        #         else:
        #             print("\n%2d 迁移学习/半监督 ..." % i)
        #             X = np.row_stack([X_train, X_test])
        #             y = np.hstack([y_train, m.model.predict(X_test)])
        #
        #         m = Model(X, y, learning_rate=0.1, application='binary', metric='rmse')
        #         m.cv(feval, return_model=True)
        #     return m.model
