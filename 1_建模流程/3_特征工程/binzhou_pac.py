# coding: utf-8
import sys
import os
import re
import time
import math
import numpy as np
import pandas as pd
import sklearn
# 数据预处理
from sklearn.preprocessing import StandardScaler  # 标准化  --以列处理数据
from sklearn.preprocessing import MinMaxScaler  # 区间缩放法 --以列处理数据
from sklearn.preprocessing import Normalizer  # 归一化 --以行处理数据
from sklearn.preprocessing import Binarizer # 二值化
from sklearn.preprocessing import LabelEncoder #简单来说 LabelEncoder 是对不连续的数字或者文本进行编号
from sklearn.preprocessing import OneHotEncoder # 和pd.get_dummies差不多
from sklearn.preprocessing import Imputer # 缺失值填充（对列操作）
from sklearn.preprocessing import PolynomialFeatures # 数据变换
from numpy import log1p
from sklearn.preprocessing import FunctionTransformer
# 特征选择 
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
from sklearn.feature_selection import chi2
from minepy import MINE # 进行互信息计算的，这个包不一定有得安装
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# print classification_report(pred, y_test_label)
from sklearn.cross_validation import train_test_split
from sklearn.externals import joblib # 保存和读取模型

# 特征工程
#################################################################
'''
目录
1 特征工程是什么？
2 数据预处理
  2.1 无量纲化　
    2.1.1 标准化　
    2.1.2 区间缩放法　
    2.1.3 标准化与归一化的区别　　
  2.2 对定量特征二值化　　
  2.3 对定性特征哑编码　
  2.4 缺失值计算　　
  2.5 数据变换
3 特征选择
  3.1 Filter
    3.1.1 方差选择法　　　　
    3.1.2 相关系数法
    3.1.3 卡方检验　
    3.1.4 互信息法　　
    3.1.5 IV （常用）
  3.2 Wrapper　
    3.2.1 递归特征消除法　　
  3.3 Embedded
    3.3.1 基于惩罚项的特征选择法 (常用)　　　
    3.3.2 基于树模型的特征选择法 (常用）
4 降维
  4.1 主成分分析法（PCA）　　
  4.2 线性判别分析法（LDA）
5 总结
'''
###############################################################
# pd.DataFrame to np.ndarray


class Feature_Engineering(object):

	'''
	 初始化FE
	'''
	def __init__(self):
		pass

	# 2 数据预处理
	# 标准化  
	def stdScalar_np(self, X):
		# 注：返回的是numpy.ndarray
		return StandardScaler().fit_transform(X)
	def stdScalar_pd(self, X):
		col_list = X.columns
		std_data = StandardScaler().fit_transform(X)
		return pd.DataFrame(std_data, columns=col_list)

	# 区间缩放法  
	def mmScalar_np(self, X):
		# 注：返回的是numpy.ndarray
		return MinMaxScaler().fit_transform(X)
	def mmScalar_pd(self, X):
		col_list = X.columns
		mm_data = MinMaxScaler().fit_transform(X)
		return pd.DataFrame(mm_data, columns=col_list)

	# 归一化
	def normScalar_np(self, X):
		# 注：返回的是numpy.ndarray
		return Normalizer().fit_transform(X)
	def normScalar_pd(self, X):
		col_list = X.columns
		norm_data = Normalizer().fit_transform(X)
		return pd.DataFrame(norm_data, columns=col_list)

	# 二值化
	def binarizer_np(self, X, thres=3):
		# 注：返回的是numpy.ndarray
		return Binarizer(threshold=thres).fit_transform(X)
	def binarizer_pd(self, X, thres=3):
		col_list = X.columns
		ts_data = Binarizer(threshold=thres).fit_transform(X)
		return pd.DataFrame(ts_data, columns=col_list)

	# 连续性变量多值化（即分箱操作，先拟合DT然后根据分支结点进行分段的选择）
	# 后续填充一下代码

	# 对于离散型变量可以进行一个自创的操作，根据label 0和1对应的类别算一个比例作为新特征
	# 这里输入的col_list的列一般IV值都比较高，新的列为数值型可以进行后续的polynomial_transform操作
	def bz_feature_create_m1(self, X_train1, y_train1, col_list, X_test_final1 = None, test_on = False):

	    X_train = X_train1.copy() # deep copy
	    y_train = y_train1.copy()
	    if X_test_final1 is not None:
	        X_test_final = X_test_final1.copy()
	    
	    for feature in col_list:

	        train_feature_add = []
	        train_feature_name = feature + '_create_m1'
	        test_feture_add = []
	        test_feature_name = feature + '_create_m1'

	        tmp_df1 = pd.DataFrame({'new_f':X_train[feature], 'label':y_train})
	        tmp_df2 = tmp_df1[tmp_df1.label == 0].new_f.value_counts() * 1.0 / tmp_df1[tmp_df1.label == 1].new_f.value_counts()
	        tmp_df3 = pd.DataFrame({'class_f' :tmp_df2.index, 'value': tmp_df2.values})
	        for i in X_train[feature]:
	            train_feature_add.append(tmp_df3.value[tmp_df3.class_f == i].iloc[0])
	        X_train[train_feature_name] = train_feature_add
	        if test_on:
	            for j in X_test_final[feature]:
	                test_feture_add.append(tmp_df3.value[tmp_df3.class_f == j].iloc[0])
	            X_test_final[test_feature_name] = test_feture_add
	    if test_on:
	        return X_train, X_test_final
	    return X_train


	# 对定性特征进行独热编码 pd.get_dummies(这个优点是方便快速，缺点是比如train中性别有男女，而test中只有男则会报错)
	# ------
	# 输入的type_d有dataframe，series，array 这个onehot用的是pd.get_dummies
        # 【注：性别以及其他的只有两个类别的feature不用onehot(哑元)，大于等于三类的需要！！！】
        # 这里的col_name是一个list，可包含一个或者多个列名比如['BusinessTravel']或者['BusinessTravel',EducationType']
	def onehot(self, X, col_name, remove_org_col=True):
		data_tmp = X.copy()
		for feature in col_name:
			if len(set(data_tmp[feature])) <= 2:
				data_tmp[feature] = pd.get_dummies(data_tmp[feature]).iloc[:, 0]  # 如果指标中只有两类则直接转成0和1,，不分成两列
			else:
				ohe = None
				ohe = pd.get_dummies(data_tmp[feature])  # 展成一个dataframe
				for new_index_name in ohe.columns:
					data_tmp[feature + '_' + new_index_name] = ohe[new_index_name]
				if remove_org_col:
					del data_tmp[feature]
		# 返回的是dataframe
		return data_tmp
	# ------
	# 这个onehot用的是sklearn.preprocessing.OneHotEncoder,兼容性更好，可以填补pd.get_dummies的缺点,但不能
	# 处理test中多一个类别的这种情况，只能少不能多。
	# 在进行OneHotEncoder之前，类别型(字符型)都得转换成数值型
	def onehot_compat(self, array, istring = False, name = 'default'):
		
		if istring:
			le = LabelEncoder()
			bt_model = le.fit(array)
			# 把bt_model变成全局变量，或者保存一下bit_model, 这样就可以直接transform test中的特征了
			# 如果不保存模型则直接 le.fit_transform(array)
			array = bt_model.transform(array)
		# 不保存模型
		new_df = pd.DataFrame(OneHotEncoder(sparse=False).fit_transform(pd.DataFrame(array))) # 或者array.reshape(-1,1)
		feature_l = []
		for index in xrange(new_df.shape[1]):
			feature_l.append(name + str(index))
		new_df.columns = feature_l
		return new_df

	# 2.4 缺失值计算 (针对连续性变量) 1.均值填充或者类别均值填充（某一个类别下的均值）2.其他的方法比如直接跑DT or RF因为允许缺失值存在
	# 3.预测缺失值（RF）
	
	def fill_na(self, X, col_list, stragegy = 'mean'):
		# 
		for feature in col_list:
			X[feature] = Imputer(stragegy=stragegy).fit_transform(X[feature])
		return X

	# 2.5(1) 数据变换 polynomial
	def poly_transform(self, X, col_list, name = 'poly2_d'):

		partial_data = X.loc[:,col_list]
		poly_data = pd.DataFrame(PolynomialFeatures(degree=2,interaction_only=False).fit_transform(partial_data))
		poly_data = poly_data.iloc[:,len(col_list)+1:]
		feature_l = []
		for index in xrange(poly_data.shape[1]):
			feature_l.append(name + str(index))
		poly_data.columns = feature_l
		X = X.join(poly_data)
		return X

	# 2.5(2) 数据变换--log变换（原来的变量会被替代）
	def log_transform(self, X, col_list):
		col_org = X.columns
		log_col_list = []
		for name in col_org:
			if name in set(col_list):
				log_col_list.append('log_'+name)
			else:
				log_col_list.append(name)
		# log_col_list = [ '_' + i for i in col_list]
		for feature in col_list:
			X[feature] = FunctionTransformer(log1p).fit_transform(X[feature])
		X.columns = log_col_list
		return X

	# 3 特征选择 数据必须一列列放进去，整体放进去的话选择完特征以后不知道对应的特征名是什么了
	# 3.1 Filter
	# 方差选择法 返回keep or remove 列名
	def var_select(self, X, one_numeric_col_name, threshold = 3):
		try:
			VarianceThreshold(threshold=threshold).fit_transform(X[[one_numeric_col_name]])
			# X[one_numeric_col_name]这个得到的是Series，X[[one_numeric_col_name]]得到的是一列的DataFrame
		except Exception as e:
			return 'remove ' + one_numeric_col_name
		return 'keep ' + one_numeric_col_name

	# 相关系数法和卡方检验返回的是p-value，越接近0越好， 互信息法返回的值越大越好
	# 可输入的方法暂时有皮尔逊相关系数、卡方检验、互信息法
	def col_select(self, X, y, one_numeric_col_name, method = 'pearsonr'):
		if method == 'pearsonr':
			return pearsonr(X[one_numeric_col_name], y)[1] # 返回的是p-value
		if method == 'chi2':
			return chi2(X[[one_numeric_col_name]], y)[1][0]
		if method == 'MINE':
			return self.__mic(X[one_numeric_col_name], y)
	# __mic是私有函数，只能被内部类调用，外部不能访问
	def __mic(self, x, y):
		m = MINE()
		m.compute_score(x, y)
		return m.mic()

	# WOE, IV(information value) 来选取特征重要性(用IV去衡量变量预测能力)
	# 计算一列的IV值  IV = (py - pn)*WOE

	def IV_cal(self, feature, label, truc = 10):
	    
	    if len(set(feature)) >= truc:
	        return 'Not Discrete, please Bin it!'
	    
	    feature2 = pd.DataFrame({'feature':feature,'label':label})
	    P = len(feature2[feature2.label==1])
	    N = len(feature2[feature2.label==0])
	    l2 = []
	    for feature_sub in set(feature):
	        df_sub = None
	        df_sub = feature2[feature2.feature == feature_sub]
	        pi = 0
	        ni = 0
	        pi = len(df_sub[df_sub.label==1])
	        ni = len(df_sub[df_sub.label==0])
	        l2.append((pi*1.0/P - ni*1.0/N) * math.log((pi * 1.0/P) / (ni*1.0/N)))
	    return sum(l2)

	# 3.2 Wrapper
	# 递归特征消除法（不常用，略）

	# 3.3 Embedded
	# 基于GBDT，RF，LR的特征选择法（常用） LR暂时不考虑
	def model_based_select(self, X, y, base_model = 'RF', seed = 1001):
		if base_model == 'RF':
			rf = RandomForestClassifier(random_state=seed).fit(X, y)
			feature_importance = rf.feature_importances_
		if base_model == 'GBDT':
			gbdt = GradientBoostingClassifier(random_state=seed).fit(X, y)
			feature_importance = gbdt.feature_importances_
		df = pd.DataFrame({'feature':X.columns, 'feature_importance':feature_importance})
		df = df.sort_values(['feature_importance'],ascending = False)
		return df

	# 4 降维 主成分分析和线性判别分析法
	def decomp(self, X, y, method = 'PCA', n_components = 2):
		if method == 'PCA':
			return PCA(n_components=n_components).fit_transform(X)
		if method == 'LDA':
			return LDA(n_components=n_components).fit_transform(X, y)


# 调参
# 待补充

# 训练测试样本集 stratify可以指定分割是否需要分层，分层的话正负样本在分割后还是保持一致, 输入的label
def train_test_sep(X, test_size = 0.25, stratify = None, random_state = 1001):
	train, test = train_test_split(X, test_size = test_size, stratify = stratify, random_state = random_state)
	return train, test
# 保存和读取模型
def save_model(model, path, model_name):
	save_path = path + model_name
	joblib.dump(model, save_path)
def load_model(path, model_name):
	load_path = path + model_name
	model = joblib.load(load_path)
	return model
