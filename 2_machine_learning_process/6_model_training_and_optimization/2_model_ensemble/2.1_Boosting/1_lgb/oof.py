#!/usr/bin/env python
# -*- coding: utf-8 -*-

############ 二分类 ##############
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from tqdm import tqdm

df = pd.read_csv('final_data_1.csv')
df['null_num_1'] = df.isnull().sum(1)

for i in tqdm(range(2, 13)):
    df_ = pd.read_csv('final_data_%s.csv' % i)
    df_[f'null_num_{i}'] = df_.isnull().sum(1)
    df = pd.merge(df, df_, on='user_id')

df['null_num_all'] = df.isnull().sum(1)

train = pd.concat(
    (pd.read_csv('train_accept_label.csv'), pd.read_csv('train_reject_label.csv'), pd.read_csv('test_stage1.csv')))
data = pd.merge(df, train, 'left', 'user_id')

cat_feats = ['feat_1',
             'feat_2',
             'feat_3',
             'feat_4',
             'feat_5',
             'feat_6',
             'feat_7',
             'feat_8',
             'feat_10',
             'feat_12',
             'feat_13',
             'feat_14',
             'feat_15',
             'feat_16',
             'feat_18',
             'feat_54',
             'feat_55',
             'feat_769',
             'feat_772',
             'feat_773',
             'feat_894',
             'feat_897']

for i in tqdm(cat_feats):
    data['c_' + i] = data[i].map(data[i].value_counts())
    data['r_' + i] = data[i].rank()


def train_model(X, y, X_test, cv, cv_seed, lgb_seed):
    params = {'boosting_type': 'gbdt',  # 'rf', 'dart', 'goss'
              'objective': 'binary',
              # 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
              'max_depth': 7,
              'n_estimators': 3000,
              'num_leaves': 63,  # 根据具体问题调整
              'learning_rate': 0.02421306293966501,
              'min_split_gain': 0.0,
              'min_child_weight': 19.75111813614344,
              'min_child_samples': 20,
              'subsample': 0.57,
              'subsample_freq': 8,
              'colsample_bytree': 0.1999,
              'reg_alpha': 5.455077557037526,
              'reg_lambda': 5.8700819100907715,
              'scale_pos_weight': 1,
              'random_state': lgb_seed,
              'n_jobs': 32}
    clf = LGBMClassifier(**params)

    oof_preds = np.zeros(X.shape[0])
    sub_preds = np.zeros((X_test.shape[0], cv))
    for n_fold, (train_idx, valid_idx) in enumerate(StratifiedKFold(cv, True, cv_seed).split(X, y), 1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_valid, y_valid = X[valid_idx], y[valid_idx]

        clf.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_valid, y_valid)],
                eval_metric='auc',
                early_stopping_rounds=300,
                verbose=0)
        oof_preds[valid_idx] = clf.predict_proba(X_valid)[:, 1]
        sub_preds[:, n_fold - 1] = clf.predict_proba(X_test)[:, 1]
    sub_preds = sub_preds.mean(1)
    print('TRIN AUC:', roc_auc_score(y, oof_preds))
    return sub_preds


X = data[data.label.isin((0, 1))].drop(['user_id', 'label'], 1).values
y = data[data.label.isin((0, 1))].label.values
X_test = data[data.label.isnull()]

rst = 0
for cv_seed in range(10):
    for lgb_seed in range(36):
        rst += train_model(X, y, X_test, 5, cv_seed, lgb_seed) / 360
