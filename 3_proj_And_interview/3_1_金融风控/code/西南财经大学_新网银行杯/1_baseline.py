class FeatureSelector():

    def __init__(self, data, labels=None):

        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')

        self.base_features = list(data.columns)
        self.one_hot_features = None

        # Dataframes recording information about features to remove
        self.record_missing = None
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        self.missing_stats = None
        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None

        # Dictionary to hold removal operations
        self.ops = {}

        self.one_hot_correlated = False


    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(
            columns={'index': 'feature',
                     0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_collinear(self, correlation_threshold, one_hot=False):

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot

        # Calculate the correlations between every column
        if one_hot:

            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

            corr_matrix = pd.get_dummies(features).corr()

        else:
            print('Compute Corr Matrix ...')
            corr_matrix = self.data.corr()

        self.corr_matrix = corr_matrix

        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns=['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in tqdm(to_drop, 'correlated features'):
            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                              'corr_feature': corr_features,
                                              'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index=True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop

        print('%d features with a correlation magnitude greater than %0.2f.\n' % (
            len(self.ops['collinear']), self.correlation_threshold))


def find_corr_pairs_sub(train_x, train_y, eps=0.01):
    feature_size = train_x.shape[1]
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print(i)
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] - train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                feature_corr_list.append(feature_corr)
    
    return feature_corr_list

def find_corr_pairs_plus(train_x, train_y, eps=0.01):
    feature_size = train_x.shape[1]
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print(i)
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] + train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, corr[0])
                feature_corr_list.append(feature_corr)
    return feature_corr_list

def find_corr_pairs_mul(train_x, train_y,eps=0.01):
    feature_size = train_x.shape[1]
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print(i)
        for j in range(feature_size):
            if i < j:
                corr = stats.pearsonr(train_x[:,i] * train_x[:,j], train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (i, j, abs(corr[0]))
                feature_corr_list.append(feature_corr)
    
    return feature_corr_list

def find_corr_pairs_divide(train_x, train_y, eps=0.01):
    feature_size = train_x.shape[1]
    feature_corr_list = []
    for i in range(feature_size):
        if i % 50 == 0:
            print(i)
        for j in range(feature_size):
            if i != j:
                try:
                    res = train_x[:,i] / train_x[:,j]
                    corr = stats.pearsonr(res, train_y)
                    if abs(corr[0]) < eps:
                        continue
                    feature_corr = (i, j, abs(corr[0]))
                    feature_corr_list.append(feature_corr)
                except ValueError:
                    print('divide 0')
    
    return feature_corr_list

def find_corr_pairs_sub_mul(train_x, train_y, sorted_corr_sub, eps=0.01):
    feature_size = train_x.shape[1]
    feature_corr_list = []
    for i in range(len(sorted_corr_sub)):
        ind_i = sorted_corr_sub[i][0]
        ind_j = sorted_corr_sub[i][1]
        if i % 100 == 0:
            print(i)
        for j in range(feature_size):
            if j != ind_i and j != ind_j :
                res = (train_x[:,ind_i] - train_x[:, ind_j]) * train_x[:,j]
                corr = stats.pearsonr(res, train_y)
                if abs(corr[0]) < eps:
                    continue
                feature_corr = (ind_i, ind_j, j, corr[0])
                feature_corr_list.append(feature_corr)
    return feature_corr_list

def get_distinct_feature_pairs(sorted_corr_list):
    distinct_list = []
    dis_ind = {}
    for i in range(len(sorted_corr_list)):
        if sorted_corr_list[i][0] not in dis_ind and sorted_corr_list[i][1] not in dis_ind:
            dis_ind[sorted_corr_list[i][0]] = 1
            dis_ind[sorted_corr_list[i][1]] = 1
            distinct_list.append(sorted_corr_list[i])
    return distinct_list

def get_distinct_feature_pairs2(sorted_corr_list):
    distinct_list = []
    dis_ind = {}
    for sorted_corr in sorted_corr_list:
        cnt = 0
        for i in range(3):
            if sorted_corr[i] in dis_ind:
                cnt = cnt + 1
        if cnt > 1:
            continue
        for i in range(3):
            dis_ind[sorted_corr[i]] = 1
        distinct_list.append(sorted_corr)
    return distinct_list

def get_feature_pair_sub_list(train_x, train_y, eps=0.01):
    sub_list = find_corr_pairs_sub(train_x, train_y, eps)
    sub_list2 = [corr for corr in sub_list if abs(corr[2])>eps]
    sorted_sub_list = sorted(sub_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_sub_list = get_distinct_feature_pairs(sorted_sub_list)
    feature_pair_sub_list = [[corr[0],corr[1]] for corr in feature_pair_sub_list]
    return feature_pair_sub_list

def get_feature_pair_plus_list(train_x, train_y, eps=0.01):
    plus_list = find_corr_pairs_plus(train_x, train_y, eps)
    plus_list2 = [corr for corr in plus_list if abs(corr[2])>eps]
    sorted_plus_list = sorted(plus_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_plus_list = get_distinct_feature_pairs(sorted_plus_list)
    feature_pair_plus_list = [[corr[0],corr[1]] for corr in feature_pair_plus_list]
    return feature_pair_plus_list

def get_feature_pair_mul_list(train_x, train_y, eps=0.01):
    mul_list = find_corr_pairs_mul(train_x, train_y, eps)
    mul_list2 = [corr for corr in mul_list if abs(corr[2])>eps]
    sorted_mul_list = sorted(mul_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_mul_list = get_distinct_feature_pairs(sorted_mul_list)
    feature_pair_mul_list = [[corr[0],corr[1]] for corr in feature_pair_mul_list]
    return feature_pair_mul_list

def get_feature_pair_divide_list(train_x, train_y, eps=0.01):
    divide_list = find_corr_pairs_divide(train_x, train_y, eps)
    divide_list2 = [corr for corr in divide_list if abs(corr[2])>eps]
    sorted_divide_list = sorted(divide_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_divide_list = get_distinct_feature_pairs(sorted_divide_list)
    feature_pair_divide_list = [[corr[0],corr[1]] for corr in feature_pair_divide_list]
    return feature_pair_divide_list

def get_feature_pair_sub_mul_list(train_x, train_y, eps=0.01):
    feature_pair_sub_list = get_feature_pair_sub_list(train_x, train_y, eps=0.01)
    sub_mul_list = find_corr_pairs_sub_mul(train_x, train_y, feature_pair_sub_list, eps=0.01)
    sub_mul_list2 = [corr for corr in sub_mul_list if abs(corr[3]) > eps]
    sorted_sub_mul_list = sorted(sub_mul_list2, key=lambda corr:abs(corr[2]), reverse=True)
    feature_pair_sub_mul_list = get_distinct_feature_pairs2(sorted_sub_mul_list)
    feature_pair_sub_mul_list = [[corr[0], corr[1], corr[2]] for corr in feature_pair_sub_mul_list]
    return feature_pair_sub_mul_list

def get_data_forDF(train_x, num_features, feature_minus_pair_list:list, feature_plus_pair_list:list,feature_mul_pair_list:list,\
               feature_divide_pair_list:list,feature_pair_sub_mul_list:list):
    # sub
    for i in range(len(feature_minus_pair_list)):
        f1 = feature_minus_pair_list[i][0]
        f2 = feature_minus_pair_list[i][1]
        train_x[f1+'_'+f2+'_sub'] = train_x[f1] - train_x[f2]
    # plus
    for i in range(len(feature_plus_pair_list)):
        f1 = feature_plus_pair_list[i][0]
        f2 = feature_plus_pair_list[i][1]
        train_x[f1+'_'+f2+'_plus'] = train_x[f1] + train_x[f2]
    # mul
    for i in range(len(feature_mul_pair_list)):
        f1 = feature_mul_pair_list[i][0]
        f2 = feature_mul_pair_list[i][1]
        train_x[f1+'_'+f2+'_mul'] = train_x[f1] * train_x[f2]
    # div
    for i in range(len(feature_divide_pair_list)):
        f1 = feature_divide_pair_list[i][0]
        f2 = feature_divide_pair_list[i][1]
        train_x[f1+'_'+f2+'_div'] = train_x[f1]*1.0 / train_x[f2]
    # sub_mul
    for i in range(len(feature_pair_sub_mul_list)):
        f1 = feature_pair_sub_mul_list[i][0]
        f2 = feature_pair_sub_mul_list[i][1]
        f3 = feature_pair_sub_mul_list[i][2]
        train_x[f1+'_'+f2+'_sub_mul'] = (train_x[f1]-train_x[f2])*train_x[f3]
    
    return train_x


#------------------------------------------------------------------------
def run(num_leaves:int):

    train1 = pd.read_csv('data/train_xy.csv')
    train2 = pd.read_csv('data/train_x.csv')
    test = pd.read_csv('data/test_all.csv')
    # 把train1和train2和test先拼接起来
    train2['y'] = -1
    test['y'] = -2
    data = pd.concat([train1, train2, test],axis=0,sort=False)
    data.replace(-99,value=np.nan,inplace=True)
    
    #-------------------------统计行缺失值状态
    # 数值型
    num_features = ['x_{}'.format(i) for i in range(1,97)]
    # 类别型
    cate_features = ['x_{}'.format(i) for i in range(97,158)]
    data['num_na_count'] = data[num_features].apply(lambda x : x.isnull().sum(), axis=1)
    data['cate_na_count'] = data[cate_features].apply(lambda x : x.isnull().sum(), axis=1)
    data['na_count'] = data['num_na_count'] + data['cate_na_count']
    #---------------------------end
    
    na_larger95_features = []
    for i in data.columns:
        if data[i].isnull().sum() / len(data) > 0.95:
            na_larger95_features.append(i)

    data.drop(na_larger95_features,axis=1,inplace=True)

    data1 = data.copy()
    d_ = {'group_1':1,'group_2':2,'group_3':3}
    data1['cust_group'] = data1['cust_group'].apply(lambda x : d_.get(x))
    # 连续型变量
    f3_8 = ['x_3','x_4','x_5','x_6','x_7','x_8']
    f9_14 = ['x_9','x_10','x_11','x_12','x_13','x_14']
    f15_20 = ['x_15','x_16','x_17','x_18','x_19','x_20']
    f21_24 = ['x_21','x_22','x_23','x_24']
    f25_30 = ['x_25','x_26','x_27','x_28','x_29','x_30']
    f31_38 = ['x_31','x_32','x_33','x_34','x_35','x_36','x_37','x_38']
    f39_46 = ['x_39','x_40','x_41','x_42','x_43','x_44','x_45','x_46']
    f47_78 = ['x_{}'.format(x) for x in range(47,79)] # optional
    f82_89 = ['x_82','x_83','x_84','x_85','x_86','x_87','x_88','x_89']
    f38_1520_2530 = f3_8 + f15_20 + f25_30
    f914_2124_3138 = f9_14 + f21_24 + f31_38
    # 离散型变量暂时不处理，就onehot
    cate_features = ['x_96' ,'x_97' ,'x_98' ,'x_99' ,'x_100','x_101','x_139','x_140','x_141','x_142','x_143','x_144',\
     'x_145','x_146','x_147','x_148','x_149','x_150','x_151','x_152','x_153','x_154','x_155','x_156','x_157']
    num_features = [x for x in data.columns if x not in cate_features+['cust_id','cust_group','y']]

    data1[cate_features] = data1[cate_features].fillna(-99)

    data1['f3_8'] = data1[f3_8].sum(axis=1,skipna=False)
    data1['f9_14'] = data1[f9_14].sum(axis=1,skipna=False)
    data1['f15_20'] = data1[f15_20].sum(axis=1,skipna=False)
    data1['f21_24'] = data1[f21_24].sum(axis=1,skipna=False)
    data1['f25_30'] = data1[f25_30].sum(axis=1,skipna=False)
    data1['f31_38'] = data1[f31_38].sum(axis=1,skipna=False)
    data1['f39_46'] = data1[f39_46].sum(axis=1,skipna=False)
    data1['f47_78'] = data1[f47_78].sum(axis=1,skipna=False)
    data1['f82_89'] = data1[f82_89].sum(axis=1,skipna=False)
    data1['f38_1520_2530'] = data1[f38_1520_2530].sum(axis=1,skipna=False)
    data1['f914_2124_3138'] = data1[f914_2124_3138].sum(axis=1,skipna=False)

    na_larger_l = ['x_93','f3_8','f9_14','f15_20','f21_24','f25_30','f31_38','f39_46','f47_78','f82_89']
    for i in na_larger_l:
        data1[i+'cate'] = data1[i].isnull() * 1

    for i in num_features + ['f3_8','f9_14','f15_20','f21_24','f25_30','f31_38','f39_46','f47_78','f82_89','f38_1520_2530','f914_2124_3138']:
        imp = Imputer()
        data1[i] = imp.fit_transform(data1[[i]]).ravel()

    # single_unique
    fs = FeatureSelector(data1[num_features])
    fs.identify_single_unique()
    data1.drop(fs.ops['single_unique'],axis=1,inplace=True)
    num_features = [x for x in num_features if x not in fs.ops['single_unique']]

    # collinear
    fs = FeatureSelector(data1[num_features])
    fs.identify_collinear(correlation_threshold=0.95)
    data1.drop(fs.ops['collinear'],axis=1,inplace=True)
    num_features = [x for x in num_features if x not in fs.ops['collinear']]

    print('特征衍生 \n')

    feature_pair_sub_list = get_feature_pair_sub_list(np.array(data1[num_features][data1.y >=0]), data1['y'][data1.y >=0])
    feature_pair_plus_list = get_feature_pair_plus_list(np.array(data1[num_features][data1.y >=0]), data1['y'][data1.y >=0])
    feature_pair_mul_list = get_feature_pair_mul_list(np.array(data1[num_features][data1.y >=0]), data1['y'][data1.y >=0])
    feature_pair_divide_list = get_feature_pair_divide_list(np.array(data1[num_features][data1.y >=0]), data1['y'][data1.y >=0])
    feature_pair_sub_mul_list = get_feature_pair_sub_mul_list(np.array(data1[num_features][data1.y >=0]), data1['y'][data1.y >=0])
    # 加工一下 feed给get_data_forDF函数
    feature_pair_sub_list = [[num_features[x[0]],num_features[x[1]]] for x in feature_pair_sub_list]
    feature_pair_plus_list = [[num_features[x[0]],num_features[x[1]]] for x in feature_pair_plus_list]
    feature_pair_mul_list = [[num_features[x[0]],num_features[x[1]]] for x in feature_pair_mul_list]
    feature_pair_divide_list = [[num_features[x[0]],num_features[x[1]]] for x in feature_pair_divide_list]
    feature_pair_sub_mul_list = [[num_features[x[0]],num_features[x[1]],num_features[x[2]]] for x in feature_pair_sub_mul_list]

    data1 = get_data_forDF(data1, num_features, feature_pair_sub_list, feature_pair_plus_list, feature_pair_mul_list,\
                     feature_pair_divide_list, feature_pair_sub_mul_list)
    
    return data1

    print('跑模型 \n')

    params = {
        'boosting': 'gbdt', # 'rf', 'dart', 'goss'
        'application': 'binary', # 'application': 'multiclass', 'num_class': 3, # multiclass=softmax, multiclassova=ova  One-vs-All
        'learning_rate': 0.01,
        'max_depth': -1,
        'num_leaves': num_leaves, # 根据具体问题调整

        'max_bin':255,
        'metric_freq':10,

        'min_split_gain': 0,
        'min_child_weight': 1,

        'bagging_fraction': 0.8,
        'feature_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 1000,
        'min_sum_hessian_in_leaf': 5.0,
        'lambda_l1': 0,
        'lambda_l2': 1,

        'scale_pos_weight': 1,
        'metric': 'auc',
        'num_threads': 40,
    }

    X = data1[data1.y>=0].drop(['cust_id','y'],axis=1)
    #     X = data1[data1.y>=0].drop(['cust_id','cust_group','y'],axis=1)
    y = data1[data1.y>=0]['y']
    lgb_data = lgb.Dataset(X, y)
    
    res = lgb.cv(
        params,
        lgb_data,
    #     categorical_feature = cate_features2,
        num_boost_round=1500,
        nfold=5,
        stratified=False,
        metrics=None,
        early_stopping_rounds=150,
        verbose_eval=50,
        show_stdv=True,
        seed=0
    )
    
    nbr = len(res['auc-mean'])
    
    clf = lgb.train(
      params,
      lgb_data,
      num_boost_round=nbr,
      valid_sets=None,
      verbose_eval=50
    )


    X = data1[data1.y==-2].drop(['cust_id','y'],axis=1)
    pred = clf.predict(X)

    res_group = data1[data1.y==-2][['cust_id']]
    res_group['pred_prob'] = pred

    return data1, res_group

if __name__ == '__main__':

    import scipy.stats as stats
    import random
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import Imputer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import KFold
    import lightgbm as lgb
    import scipy.stats as stats
    from tqdm import tqdm, tqdm_notebook

    import sys

    data = run(num_leaves=47)