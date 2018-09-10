from sklearn.neural_network import MLPClassifier
import pandas as pd
import re
import time
import datetime
import pickle
import random
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier
from numpy import log
from sklearn.metrics import roc_auc_score



def CareerYear(x):
    #对工作年限进行转换
    if x.find('n/a') > -1:
        return -1
    elif x.find("10+")>-1:   #将"10＋years"转换成 11
        return 11
    elif x.find('< 1') > -1:  #将"< 1 year"转换成 0
        return 0
    else:
        return int(re.sub("\D", "", x))   #其余数据，去掉"years"并转换成整数


def DescExisting(x):
    #将desc变量转换成有记录和无记录两种
    if type(x).__name__ == 'float':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x):
    mth_dict = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10,
                'Nov': 11, 'Dec': 12}
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))
        #time.mktime 不能读取1970年之前的日期
    else:
        yr = int(x[4:6])
        if yr <=17:
            yr = 2000+yr
        else:
            yr = 1900 + yr
        mth = mth_dict[x[:3]]
        return datetime.datetime(yr,mth,1)


def MonthGap(earlyDate, lateDate):
    if lateDate > earlyDate:
        gap = relativedelta(lateDate,earlyDate)
        yr = gap.years
        mth = gap.months
        return yr*12+mth
    else:
        return 0


def MakeupMissing(x):
    if np.isnan(x):
        return -1
    else:
        return x


def ModifyDf(x, new_value):
    if np.isnan(x):
        return new_value
    else:
        return x


folderOfData = '/Users/Code/xiaoxiang/第3期/Data/'


file1 = open(folderOfData + 'trainData.pkl', 'r')
trainData = pickle.load(file1)
file1.close()

file2 = open(folderOfData + 'testData.pkl', 'r')
testData = pickle.load(file2)
file2.close()


modelFile =open(folderOfData+'LR_Model_Normal.pkl','r')
LR = pickle.load(modelFile)
modelFile.close()

#对变量的处理只需针对入模变量即可
var_in_model = list(LR.pvalues.index)
var_in_model.remove('intercept')

file1 = open(folderOfData+'merge_bin_dict.pkl','r')
merge_bin_dict = pickle.load(file1)
file1.close()


file2 = open(folderOfData+'br_encoding_dict.pkl','r')
br_encoding_dict = pickle.load(file2)
file2.close()

file3 = open(folderOfData+'continous_merged_dict.pkl','r')
continous_merged_dict = pickle.load(file3)
file3.close()

file4 = open(folderOfData+'WOE_dict.pkl','r')
WOE_dict = pickle.load(file4)
file4.close()



'''
第一步：完成数据预处理
在实际工作中，可以只清洗模型实际使用的字段
'''

# 将带％的百分比变为浮点数
trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%',''))/100)
testData['int_rate_clean'] = testData['int_rate'].map(lambda x: float(x.replace('%',''))/100)

# 将工作年限进行转化，否则影响排序
trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)
testData['emp_length_clean'] = testData['emp_length'].map(CareerYear)

# 将desc的缺失作为一种状态，非缺失作为另一种状态
trainData['desc_clean'] = trainData['desc'].map(DescExisting)
testData['desc_clean'] = testData['desc'].map(DescExisting)

# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
testData['app_date_clean'] = testData['issue_d'].map(lambda x: ConvertDateStr(x))
testData['earliest_cr_line_clean'] = testData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))

trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x))
trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x))

# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))
testData['mths_since_last_delinq_clean'] = testData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))

trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x:MakeupMissing(x))
testData['mths_since_last_record_clean'] = testData['mths_since_last_record'].map(lambda x:MakeupMissing(x))

trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))
testData['pub_rec_bankruptcies_clean'] = testData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

'''
第二步：变量衍生
'''
# 考虑申请额度与收入的占比
trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)
testData['limit_income'] = testData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)

# 考虑earliest_cr_line到申请日期的跨度，以月份记
trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)
testData['earliest_cr_to_app'] = testData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)




for var in var_in_model:
    var1 = var.replace('_Bin_WOE','')

    # 有些取值个数少、但是需要合并的变量
    if var1 in merge_bin_dict.keys():
        print "{} need to be regrouped".format(var1)
        trainData[var1 + '_Bin'] = trainData[var1].map(merge_bin_dict[var1])
        testData[var1 + '_Bin'] = testData[var1].map(merge_bin_dict[var1])

    # 有些变量需要用bad rate进行编码
    if var1.find('_br_encoding')>-1:
        var2 =var1.replace('_br_encoding','')
        print "{} need to be encoded by bad rate".format(var2)
        trainData[var1] = trainData[var2].map(br_encoding_dict[var2])
        testData[var1] = testData[var2].map(br_encoding_dict[var2])
        #需要注意的是，有可能在测试样中某些值没有出现在训练样本中，从而无法得出对应的bad rate是多少。故可以用最坏（即最大）的bad rate进行编码
        max_br = max(testData[var1])
        testData[var1] = testData[var1].map(lambda x: ModifyDf(x, max_br))


    #上述处理后，需要加上连续型变量一起进行分箱
    if -1 not in set(testData[var1]):
        trainData[var1 + '_Bin'] = trainData[var1].map(lambda x: AssignBin(x, continous_merged_dict[var1]))
        testData[var1+'_Bin'] = testData[var1].map(lambda x: AssignBin(x, continous_merged_dict[var1]))
    else:
        trainData[var1 + '_Bin'] = trainData[var1].map(lambda x: AssignBin(x, continous_merged_dict[var1], [-1]))
        testData[var1 + '_Bin'] = testData[var1].map(lambda x: AssignBin(x, continous_merged_dict[var1],[-1]))

    #WOE编码
    var3 = var.replace('_WOE','')
    trainData[var] = trainData[var3].map(WOE_dict[var3])
    testData[var] = testData[var3].map(WOE_dict[var3])




'''
串行结构的组合模型
'''

#随机从所有的变量种选取一部分构建神经网络模型
trainData_trial = trainData.copy()
randomSelectedFeatures = random.sample(WOE_dict.keys(),20)
randomSelectedWOE = []
for var in randomSelectedFeatures:
    newVar = var+"_WOE"
    randomSelectedWOE.append(newVar)
    trainData_trial[newVar] = trainData_trial[var].map(lambda x: WOE_dict[var]['WOE'][x])


X_train = np.matrix(trainData_trial[randomSelectedWOE])
y_train = np.array(trainData_trial['y'])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
pred_prob = clf.predict_proba(X_train)[:,1]

# 神经网络的输出作为评分卡模型的输入
featureSelected  = list(LR.params.index)
featureSelected.remove('intercept')
X_scorecard = trainData[featureSelected]
X_scorecard['intercept'] = [1]*X_scorecard.shape[0]
X_scorecard['ANN'] = pred_prob


logit = sm.Logit(trainData['y'], X_scorecard)
logit_result = logit.fit()
trainData['log_odds_ensembled'] = logit_result.predict()
ks = KS(trainData, 'log_odds_ensembled', 'y')  #0.36203140834209224
auc = roc_auc_score(trainData['y'],trainData['log_odds_ensembled'])  #0.73795050743488677


'''
异构并行模型
'''
#根据上一步的结果，未调优的神经网络模型的结果是 pred_prob
X_scorecard2 = trainData[featureSelected]
X_scorecard2['intercept'] = [1]*X_scorecard.shape[0]
trainData['prob'] = LR.predict(X_scorecard2)
trainData['log_odds'] = trainData['prob'].map(lambda x: np.log(x/(1-x)))
ks = KS(trainData, 'log_odds', 'y')  #0.36021343084122087
auc = roc_auc_score(trainData['y'],trainData['log_odds']) #0.73797150195612427


trainData['prob_ANN'] = pred_prob
trainData['log_odds_ANN'] = trainData['prob_ANN'].apply(lambda x: np.log(x/(1-x)))
w_list = [i/100.0 for i in range(1,100)]
perf_list = []
for w in w_list:
    #trainData['prob_LR'] = trainData['log_odds'].apply(lambda x: 1.0/(1+np.exp(-x)))
    trainData['log_odds_combined_1'] = trainData[['log_odds_ANN','log_odds']].apply(lambda x: w*x.log_odds+(1-w)*x.log_odds_ANN,axis=1)
    auc = roc_auc_score(trainData['y'],trainData['log_odds_combined_1'])
    perf_list.append((w,auc))
best_w = max(perf_list,key=lambda x: x[1])   #(0.96, 0.73799101609445383)
best_w = best_w[0]
trainData['log_odds_combined_best'] = trainData[['log_odds_ANN','log_odds']].\
    apply(lambda x: best_w*x.log_odds+(1-best_w)*x.log_odds_ANN,axis=1)
ks = KS(trainData, 'log_odds_combined_best', 'y')  #0.36216844469625908


'''
同构并行模型。采用神经网络
'''
#Bagging
train, test = train_test_split(trainData, train_size=0.7)
total_pred = np.zeros(test.shape[0])
numberOfBagging = 10
for i in range(numberOfBagging):
    train2 = train.sample(frac = 0.6, replace = True)
    randomSelectedFeatures2 = random.sample(featureSelected,5)
    X_train = np.matrix(train2[featureSelected])
    y_train = np.array(train2['y'])
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(X_train, y_train)
    X_test = np.matrix(test[featureSelected])
    pred_prob = clf.predict_proba(X_test)[:, 1]
    pred_log_odds= np.log(pred_prob/(1-pred_prob))
    total_pred = total_pred + pred_log_odds
total_pred = total_pred/numberOfBagging
test['log_odds_bagging'] = total_pred
roc_auc_score(test['y'],test['log_odds_bagging'])  #0.73684341446752888
ks = KS(test, 'log_odds_bagging', 'y') #0.35546653400062367



X_train = np.matrix(train[featureSelected])
y_train = np.array(train['y'])
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(X_train, y_train)
X_test = np.matrix(test[featureSelected])
pred_prob = clf.predict_proba(X_test)[:,1]
log_odds_single_ANN = np.log(pred_prob/(1-pred_prob))
test['log_odds_single_ANN'] = log_odds_single_ANN
ks = KS(test, 'log_odds_single_ANN', 'y')  #0.34629475622614614
roc_auc_score(test['y'],test['log_odds_single_ANN'])  #0.73464068788333314

