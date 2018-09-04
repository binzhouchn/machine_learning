import pandas as pd
import re
import time
import datetime
from dateutil.relativedelta import relativedelta
from sklearn.model_selection import train_test_split


def CareerYear(x):
	x = str(x)
    if x.find('n') > -1:
        return -1
    elif x.find("10+")>-1:
        return 11
    elif x.find('< 1') > -1:
        return 0
    else:
        return int(re.sub("\D", "", x))


def DescExisting(x):
    x2 = str(x)
    if x2 == 'nan':
        return 'no desc'
    else:
        return 'desc'


def ConvertDateStr(x,format):
    if str(x) == 'nan':
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime('9900-1','%Y-%m')))
    else:
        return datetime.datetime.fromtimestamp(time.mktime(time.strptime(x,format)))


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



# 数据预处理
# 1，读入数据
# 2，选择合适的建模样本
# 3，数据集划分成训练集和测试集
allData = pd.read_csv('C:/Users/OkO/Desktop/Financial Data Analsys/3nd Series/Data/application.csv',header = 0)
allData['term'] = allData['term'].apply(lambda x: int(x.replace(' months','')))

# 处理标签：Fully Paid是正常用户；Charged Off是违约用户
allData['y'] = allData['loan_status'].map(lambda x: int(x == 'Charged Off'))



'''
由于存在不同的贷款期限（term），申请评分卡模型评估的违约概率必须要在统一的期限中，且不宜太长，所以选取term＝36months的行本
'''

allData1 = allData.loc[allData.term == 36]

trainData, testData = train_test_split(allData1,test_size=0.4)


'''
第一步：数据预处理，包括
（1）数据清洗
（2）格式转换
（3）确实值填补
'''




# 将带％的百分比变为浮点数
trainData['int_rate_clean'] = trainData['int_rate'].map(lambda x: float(x.replace('%',''))/100)

# 将工作年限进行转化，否则影响排序
trainData['emp_length_clean'] = trainData['emp_length'].map(CareerYear)

# 将desc的缺失作为一种状态，非缺失作为另一种状态
# trainData['desc_clean'] = trainData['desc'].map(DescExisting)
trainData['desc_clean'] = ['desc' if x else 'no desc' for x in trainData['desc'].notnull()]

# 处理日期。earliest_cr_line的格式不统一，需要统一格式且转换成python的日期
trainData['app_date_clean'] = trainData['issue_d'].map(lambda x: ConvertDateStr(x,'%b-%y'))
trainData['earliest_cr_line_clean'] = trainData['earliest_cr_line'].map(lambda x: ConvertDateStr(x,'%b-%y'))

# 处理mths_since_last_delinq。注意原始值中有0，所以用－1代替缺失
trainData['mths_since_last_delinq_clean'] = trainData['mths_since_last_delinq'].map(lambda x:MakeupMissing(x))

trainData['mths_since_last_record_clean'] = trainData['mths_since_last_record'].map(lambda x:MakeupMissing(x))

trainData['pub_rec_bankruptcies_clean'] = trainData['pub_rec_bankruptcies'].map(lambda x:MakeupMissing(x))

'''
第二步：变量衍生
'''
# 考虑申请额度与收入的占比
trainData['limit_income'] = trainData.apply(lambda x: x.loan_amnt / x.annual_inc, axis = 1)

# 考虑earliest_cr_line到申请日期的跨度，以月份记
trainData['earliest_cr_to_app'] = trainData.apply(lambda x: MonthGap(x.earliest_cr_line_clean,x.app_date_clean), axis = 1)


'''
第三步：分箱，采用ChiMerge,要求分箱完之后：
（1）不超过5箱
（2）Bad Rate单调
（3）每箱同时包含好坏样本
（4）特殊值如－1，单独成一箱

连续型变量可直接分箱
类别型变量：
（a）当取值较多时，先用bad rate编码，再用连续型分箱的方式进行分箱
（b）当取值较少时：
    （b1）如果每种类别同时包含好坏样本，无需分箱
    （b2）如果有类别只包含好坏样本的一种，需要合并
'''
num_features = ['int_rate_clean','emp_length_clean','annual_inc', 'dti', 'delinq_2yrs', 'earliest_cr_to_app','inq_last_6mths', \
                'mths_since_last_record_clean', 'mths_since_last_delinq_clean','open_acc','pub_rec','total_acc']

cat_features = ['home_ownership', 'verification_status','desc_clean', 'purpose', 'zip_code','addr_state','pub_rec_bankruptcies_clean']


more_value_features = []
less_value_features = []
# 第一步，检查类别型变量中，哪些变量取值超过5
for var in cat_features:
    valueCounts = len(set(trainData[var]))
    print valueCounts
    if valueCounts > 5:
        more_value_features.append(var)
    else:
        less_value_features.append(var)

# （i）当取值<5时：如果每种类别同时包含好坏样本，无需分箱；如果有类别只包含好坏样本的一种，需要合并
merge_bin = {}
for col in less_value_features:
    binBadRate = BinBadRate(trainData, col, 'y')[0]
    if min(binBadRate.values()) == 0 or max(binBadRate.values()) == 1:
        print '{} need to be combined'.format(col)
        combine_bin = MergeBad0(trainData, col, 'y')
        merge_bin[col] = combine_bin

# （ii）当取值>5时：用bad rate进行编码，放入连续型变量里
br_encoding_dict = {}
for col in more_value_features:
    br_encoding = BadRateEncoding(df, col, target)
    trainData[col+'_br_encoding'] = br_encoding['encoding']
    br_encoding_dict[col] = br_encoding['bad_rate']
    num_features.append(col+'_br_encoding')

# （iii）对连续型变量进行分箱，包括（ii）中的变量
for col in num_features:
    print "{} is in processing".format(col)
    if -1 not in set(trainData[col]):
        max_interval = 5
        cutOff = ChiMerge(trainData, col, target, max_interval=max_interval,special_attribute=[],minBinPcnt=0)
        trainData[col+'_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff,special_attribute=[]))
        monotone = BadRateMonotone(trainData, col+'_Bin', 'y')
        while(not monotone):
            max_interval -= 1
            cutOff = ChiMerge(trainData, col, target, max_interval=max_interval, special_attribute=[],
                                          minBinPcnt=0)
            trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[]))
            if max_interval == 2:
                # 当分箱数为2时，必然单调
                break
            monotone = BadRateMonotone(trainData, col + '_Bin', 'y')
    else:
        max_interval = 5
        cutOff = ChiMerge(trainData, col, target, max_interval=max_interval, special_attribute=[-1],
                                      minBinPcnt=0)
        trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
        monotone = BadRateMonotone(trainData, col + '_Bin', 'y')
        while (not monotone):
            max_interval -= 1
            cutOff = ChiMerge(trainData, col, target, max_interval=max_interval, special_attribute=[-1],
                                          minBinPcnt=0)
            trainData[col + '_Bin'] = trainData[col].map(lambda x: AssignBin(x, cutOff, special_attribute=[-1]))
            if max_interval == 2:
                # 当分箱数为2时，必然单调
                break
            monotone = BadRateMonotone(trainData, col + '_Bin', 'y')

