# tradaboost迁移学习

[tradaboost（更新于2021-10-11）](https://zhuanlan.zhihu.com/p/109540481)<br>


```python
import numpy as np

class Tradaboost(object):##针对二分类设计的tradaboost
    def __init__(self,N=None,base_estimator=None,threshold=None,score=roc_auc_score):    
        self.N=N
        self.threshold=threshold
        self.base_estimator=base_estimator
        self.score=score
        self.estimators=[]
            
    # 权重的标准化，其实差别不大，在前面的例子中也有用到过
    def _calculate_weights(self,weights):      
        total = np.sum(weights)      
        return np.asarray(weights / total, order='C')      
          

          
    #计算目标域上的错误率     
    def _calculate_error_rate(self,y_true, y_pred, weight):      
        total = np.sum(weight)      
        return np.sum(weight[:, 0] / total * np.abs(y_true - y_pred))      
          
    #根据逻辑回归输出的score的得到标签，注意这里不能用predict直接输出标签      
   
          
    def fit(self,source,target,source_label,target_label,early_stopping_rounds):#注意，输入要转为numpy格式的
        
        source_shape=source.shape[0]
        target_shape=target.shape[0]
        trans_data = np.concatenate((source, target), axis=0)      
        trans_label = np.concatenate((source_label,target_label), axis=0)      
        weights_source = np.ones([source_shape, 1])/source_shape      
        weights_target = np.ones([target_shape, 1])/target_shape
        weights = np.concatenate((weights_source, weights_target), axis=0)
        
        # 根据公式初始化参数，具体可见原文
        
        bata = 1 / (1 + np.sqrt(2 * np.log(source_shape / self.N)))    
        bata_T = np.zeros([1, self.N])
        result_label = np.ones([source_shape+target_shape, self.N])    

        trans_data = np.asarray(trans_data, order='C')     #行优先 
        trans_label = np.asarray(trans_label, order='C')     
        
        score=0
        flag=0
        
        for i in range(self.N):      
            P = self._calculate_weights(weights)      #权重的标准化
            self.base_estimator.fit(trans_data,trans_label,P*100)#这里xgb有bug，，如果权重系数太小貌似是被忽略掉了？
            self.estimators.append(self.base_estimator)
            y_preds=self.base_estimator.predict_proba(trans_data)[:,1] #全量数据的预测
            result_label[:, i]=y_preds #保存全量数据的预测结果用于后面的各个模型的评价
             

            #注意，仅仅计算在目标域上的错误率 ，
            y_target_pred=self.base_estimator.predict_proba(target)[:,1]#目标域的预测
            error_rate = self._calculate_error_rate(target_label, (y_target_pred>self.threshold).astype(int),  \
                                              weights[source_shape:source_shape + target_shape, :])  
            #根据不同的判断阈值来对二分类的标签进行判断，对于不均衡的数据集合很有效，比如100：1的数据集，不设置class_wegiht
            #的情况下需要将正负样本的阈值提高到99%.
            
            # 防止过拟合     
            if error_rate > 0.5:      
                error_rate = 0.5      
            if error_rate == 0:      
                N = i      
                break       

            bata_T[0, i] = error_rate / (1 - error_rate)      

            # 调整目标域样本权重      
            for j in range(target_shape):      
                weights[source_shape + j] = weights[source_shape + j] * \
                np.power(bata_T[0, i],(-np.abs(result_label[source_shape + j, i] - target_label[j])))

                
            # 调整源域样本权重      
            for j in range(source_shape):      
                weights[j] = weights[j] * np.power(bata,np.abs(result_label[j, i] - source_label[j]))
                
            tp=self.score(target_label,y_target_pred)
            print('The '+str(i)+' rounds score is '+str(tp))
            if tp > score :      
                score = tp      
                best_round = i  
                flag=0
            else:
                flag+=1
            if flag >=early_stopping_rounds:  
                print('early stop!')
                break  
        self.best_round=best_round
        self.best_score=score
```

