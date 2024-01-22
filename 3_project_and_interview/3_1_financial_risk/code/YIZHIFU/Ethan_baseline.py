# coding: utf-8
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tqdm import tqdm
import gc
warnings.filterwarnings('ignore')


def load_dataset(DATA_PATH):
    train_label = pd.read_csv(DATA_PATH+'train_label.csv')
    train_base = pd.read_csv(DATA_PATH+'train_base.csv')
    test_base = pd.read_csv(DATA_PATH+'test_a_base.csv')

    train_op = pd.read_csv(DATA_PATH+'train_op.csv')
    train_trans = pd.read_csv(DATA_PATH+'train_trans.csv')
    test_op = pd.read_csv(DATA_PATH+'test_a_op.csv')
    test_trans = pd.read_csv(DATA_PATH+'test_a_trans.csv')

    return train_label, train_base, test_base, train_op, train_trans, test_op, test_trans

def transform_time(x):
    day = int(x.split(' ')[0])
    hour = int(x.split(' ')[2].split('.')[0].split(':')[0])
    minute = int(x.split(' ')[2].split('.')[0].split(':')[1])
    second = int(x.split(' ')[2].split('.')[0].split(':')[2])
    return 86400*day+3600*hour+60*minute+second

def data_preprocess(DATA_PATH):
    train_label, train_base, test_base, train_op, train_trans, test_op, test_trans = load_dataset(DATA_PATH=DATA_PATH)
    # 拼接数据
    train_df = train_base.copy()
    test_df = test_base.copy()
    train_df = train_label.merge(train_df, on=['user'], how='left')
    del train_base, test_base

    op_df = pd.concat([train_op, test_op], axis=0, ignore_index=True)
    trans_df = pd.concat([train_trans, test_trans], axis=0, ignore_index=True)
    data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    del train_op, test_op, train_df, test_df
    # 时间维度的处理
    op_df['days_diff'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    trans_df['days_diff'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[0]))
    op_df['timestamp'] = op_df['tm_diff'].apply(lambda x: transform_time(x))
    trans_df['timestamp'] = trans_df['tm_diff'].apply(lambda x: transform_time(x))
    op_df['hour'] = op_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['hour'] = trans_df['tm_diff'].apply(lambda x: int(x.split(' ')[2].split('.')[0].split(':')[0]))
    trans_df['week'] = trans_df['days_diff'].apply(lambda x: x % 7)
    # 排序
    trans_df = trans_df.sort_values(by=['user', 'timestamp'])
    op_df = op_df.sort_values(by=['user', 'timestamp'])
    trans_df.reset_index(inplace=True, drop=True)
    op_df.reset_index(inplace=True, drop=True)

    gc.collect()
    return data, op_df, trans_df

def gen_user_amount_features(df):
    group_df = df.groupby(['user'])['amount'].agg({
        'user_amount_mean': 'mean',
        'user_amount_std': 'std',
        'user_amount_max': 'max',
        'user_amount_min': 'min',
        'user_amount_sum': 'sum',
        'user_amount_med': 'median',
        'user_amount_cnt': 'count',
        }).reset_index()
    return group_df

def gen_user_group_amount_features(df, value):
    group_df = df.pivot_table(index='user',
                              columns=value,
                              values='amount',
                              dropna=False,
                              aggfunc=['count', 'sum']).fillna(0)
    group_df.columns = ['user_{}_{}_amount_{}'.format(value, f[1], f[0]) for f in group_df.columns]
    group_df.reset_index(inplace=True)

    return group_df

def gen_user_window_amount_features(df, window):
    group_df = df[df['days_diff']>window].groupby('user')['amount'].agg({
        'user_amount_mean_{}d'.format(window): 'mean',
        'user_amount_std_{}d'.format(window): 'std',
        'user_amount_max_{}d'.format(window): 'max',
        'user_amount_min_{}d'.format(window): 'min',
        'user_amount_sum_{}d'.format(window): 'sum',
        'user_amount_med_{}d'.format(window): 'median',
        'user_amount_cnt_{}d'.format(window): 'count',
        }).reset_index()
    return group_df

def gen_user_nunique_features(df, value, prefix):
    group_df = df.groupby(['user'])[value].agg({
        'user_{}_{}_nuniq'.format(prefix, value): 'nunique'
    }).reset_index()
    return group_df

def gen_user_null_features(df, value, prefix):
    df['is_null'] = 0
    df.loc[df[value].isnull(), 'is_null'] = 1

    group_df = df.groupby(['user'])['is_null'].agg({'user_{}_{}_null_cnt'.format(prefix, value): 'sum',
                                                    'user_{}_{}_null_ratio'.format(prefix, value): 'mean'}).reset_index()
    return group_df

def gen_user_tfidf_features(df, value):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = TfidfVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_tfidf_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def gen_user_countvec_features(df, value):
    df[value] = df[value].astype(str)
    df[value].fillna('-1', inplace=True)
    group_df = df.groupby(['user']).apply(lambda x: x[value].tolist()).reset_index()
    group_df.columns = ['user', 'list']
    group_df['list'] = group_df['list'].apply(lambda x: ','.join(x))
    enc_vec = CountVectorizer()
    tfidf_vec = enc_vec.fit_transform(group_df['list'])
    svd_enc = TruncatedSVD(n_components=10, n_iter=20, random_state=2020)
    vec_svd = svd_enc.fit_transform(tfidf_vec)
    vec_svd = pd.DataFrame(vec_svd)
    vec_svd.columns = ['svd_countvec_{}_{}'.format(value, i) for i in range(10)]
    group_df = pd.concat([group_df, vec_svd], axis=1)
    del group_df['list']
    return group_df

def kfold_stats_feature(train, test, feats, k):
    '''
    函数说明：基于target的编码
    数据：用户的年龄，性别，职业等
    标签：用户是否违约？0或1
    这种情况下，我们就构造目标编码，分析不同年龄人群违约或者不违约的比例，作为特征去直接预测。不过这种情况下构造的特征很容易过拟合，因此常常和五折特征搭配使用
    '''
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)  # 这里最好和后面模型的K折交叉验证保持一致

    train['fold'] = None
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
        train.loc[val_idx, 'fold'] = fold_

    kfold_features = []
    for feat in feats:
        nums_columns = ['label']
        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            kfold_features.append(colname)
            train[colname] = None
            for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, train['label'])):
                tmp_trn = train.iloc[trn_idx]
                order_label = tmp_trn.groupby([feat])[f].mean()
                tmp = train.loc[train.fold == fold_, [feat]]
                train.loc[train.fold == fold_, colname] = tmp[feat].map(order_label)
                # fillna
                global_mean = train[f].mean()
                train.loc[train.fold == fold_, colname] = train.loc[train.fold == fold_, colname].fillna(global_mean)
            train[colname] = train[colname].astype(float)

        for f in nums_columns:
            colname = feat + '_' + f + '_kfold_mean'
            test[colname] = None
            order_label = train.groupby([feat])[f].mean()
            test[colname] = test[feat].map(order_label)
            # fillna
            global_mean = train[f].mean()
            test[colname] = test[colname].fillna(global_mean)
            test[colname] = test[colname].astype(float)
    del train['fold']
    return train, test

def gen_features(df, op, trans):
    df.drop(['service3_level'], axis=1, inplace=True)
    # base
    df['product7_fail_ratio'] = df['product7_fail_cnt'] / df['product7_cnt']
    df['city_count'] = df.groupby(['city'])['user'].transform('count')
    df['province_count'] = df.groupby(['province'])['user'].transform('count')
    # trans
    df = df.merge(gen_user_amount_features(trans), on=['user'], how='left')
    for col in tqdm(['days_diff', 'platform', 'tunnel_in', 'tunnel_out', 'type1', 'type2', 'ip', 'ip_3']):
        df = df.merge(gen_user_nunique_features(df=trans, value=col, prefix='trans'), on=['user'], how='left')
    df['user_amount_per_days'] = df['user_amount_sum'] / df['user_trans_days_diff_nuniq']
    df['user_amount_per_cnt'] = df['user_amount_sum'] / df['user_amount_cnt']
    df = df.merge(gen_user_group_amount_features(df=trans, value='platform'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type1'), on=['user'], how='left')
    df = df.merge(gen_user_group_amount_features(df=trans, value='type2'), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=27), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=23), on=['user'], how='left')
    df = df.merge(gen_user_window_amount_features(df=trans, window=15), on=['user'], how='left')
    df = df.merge(gen_user_null_features(df=trans, value='ip', prefix='trans'), on=['user'], how='left')
    group_df = trans[trans['type1']=='45a1168437c708ff'].groupby(['user'])['days_diff'].agg({'user_type1_45a1168437c708ff_min_day': 'min'}).reset_index()
    df = df.merge(group_df, on=['user'], how='left')
    # op
    df = df.merge(gen_user_tfidf_features(df=op, value='op_mode'), on=['user'], how='left')
    df = df.merge(gen_user_tfidf_features(df=op, value='op_type'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_mode'), on=['user'], how='left')
    df = df.merge(gen_user_countvec_features(df=op, value='op_type'), on=['user'], how='left')
    # LabelEncoder
    cat_cols = []
    for col in tqdm([f for f in df.select_dtypes('object').columns if f not in ['user']]):
        le = LabelEncoder()
        df[col].fillna('-1', inplace=True)
        df[col] = le.fit_transform(df[col])
        cat_cols.append(col)

    return df

def lgb_model(train, target, test, k):
    feats = [f for f in train.columns if f not in ['user', 'label']]
    print('Current num of features:', len(feats))
    folds = StratifiedKFold(n_splits=k, shuffle=True, random_state=2020)
    oof_probs = np.zeros(train.shape[0])
    output_preds = 0
    offline_score = []
    feature_importance_df = pd.DataFrame()
    parameters = {
        'learning_rate': 0.05,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'num_leaves': 63,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'nthread': 8
    }

    for i, (train_index, test_index) in enumerate(folds.split(train, target)):
        train_y, test_y = target[train_index], target[test_index]
        train_X, test_X = train[feats].iloc[train_index, :], train[feats].iloc[test_index, :]

        dtrain = lgb.Dataset(train_X,
                             label=train_y)
        dval = lgb.Dataset(test_X,
                           label=test_y)
        lgb_model = lgb.train(
                parameters,
                dtrain,
                num_boost_round=5000,
                valid_sets=[dval],
                early_stopping_rounds=100,
                verbose_eval=100,
        )
        oof_probs[test_index] = lgb_model.predict(test_X[feats], num_iteration=lgb_model.best_iteration)
        offline_score.append(lgb_model.best_score['valid_0']['auc'])
        output_preds += lgb_model.predict(test[feats], num_iteration=lgb_model.best_iteration)/folds.n_splits
        print(offline_score)
        # feature importance
        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = lgb_model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))
    print('feature importance:')
    print(feature_importance_df.groupby(['feature'])['importance'].mean().sort_values(ascending=False).head(15))

    return output_preds, oof_probs, np.mean(offline_score)

if __name__ == '__main__':
    DATA_PATH = 'data/'
    print('读取数据...')
    data, op_df, trans_df = data_preprocess(DATA_PATH=DATA_PATH)

    print('开始特征工程...')
    data = gen_features(data, op_df, trans_df)
    data['city_level'] = data['city'].map(str) + '_' + data['level'].map(str)
    data['city_balance_avg'] = data['city'].map(str) + '_' + data['balance_avg'].map(str)

    print('开始模型训练...')
    train = data[~data['label'].isnull()].copy()
    target = train['label']
    test = data[data['label'].isnull()].copy()

    target_encode_cols = ['province', 'city', 'city_level', 'city_balance_avg']
    train, test = kfold_stats_feature(train, test, target_encode_cols, 5)
    train.drop(['city_level', 'city_balance_avg'], axis=1, inplace=True)
    test.drop(['city_level', 'city_balance_avg'], axis=1, inplace=True)

    lgb_preds, lgb_oof, lgb_score = lgb_model(train=train, target=target, test=test, k=5)

    sub_df = test[['user']].copy()
    sub_df['prob'] = lgb_preds
    sub_df.to_csv('sub.csv', index=False)