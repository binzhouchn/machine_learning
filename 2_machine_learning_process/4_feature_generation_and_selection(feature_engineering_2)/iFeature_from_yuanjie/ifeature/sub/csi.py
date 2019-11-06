# coding=utf-8

__author__ = 'binzhou'
__mtime__ = '20181001'


# 一般用csi就行了，下面的var_csi是dataframe
ef csi(actual,expect,val,q):
    var_cat=pd.qcut(actual[val].values,q=q,duplicates='drop')
    var_cat_cnt=var_cat.value_counts()+0.0001
    var_cnt_proportion=var_cat_cnt/sum(var_cat_cnt)
    age_pred_cnt=pd.cut(expect[val].values,bins=var_cat.categories).value_counts()
    age_pred_cnt_1=age_pred_cnt+0.0001
    age_pred_proportion=age_pred_cnt_1/sum(age_pred_cnt_1)
    return sum((var_cnt_proportion-age_pred_proportion)*np.log(var_cnt_proportion/age_pred_proportion))

def var_csi(actual,expect,var_name,q=10):
    result=[]
    for i in var_name:
        #rint(i)
        result.append(csi(actual,expect,i,q=q))
    return pd.DataFrame(result,index=var_name,columns=['csi'])