import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import time
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel,delayed
def loss_function(y, y_pred):
    """
    定义损失函数，这里假设是平方损失
    """
    return np.sum((y - y_pred) ** 2)
def CEloss_function(y, y_pred):##
    """
    定义损失函数，这里假设是平方损失
    """
    return np.sum(-np.log(y_pred))
def objective_function(gamma, y, F_prev, h):
    """
    定义优化目标函数
    """
    y_pred = F_prev + gamma * h
    return CEloss_function(y, y_pred)

def update_model(y, F_prev, h):
    """
    更新模型参数
    """
    result = minimize(
        lambda gamma: objective_function(gamma, y, F_prev, h),
        x0=0.0,  # 初始猜测值
        method='L-BFGS-B'  # 选择适当的优化方法
    )
    gamma_optimal = result.x[0]
    return gamma_optimal

# # 示例用法
# # 假设你有训练数据 X, y 和上一轮的预测 F_prev，以及一个新的弱学习器 h
# X = np.random.rand(100, 10)
# y = np.random.randint(0, 2, size=100)
# F_prev = np.random.rand(100)
# h = np.random.rand(100)

# # 更新模型
# gamma_optimal = update_model(X, y, F_prev, h)


def check_classnum(idx,y):
    if len(np.unique(y[idx])) !=len(np.unique(y)) :
        return False
    else:
        return np.all(np.equal(np.unique(y[idx]),np.unique(y)))
from utils import splitting_predict_proba
class SamplingGradientboost_noweight():
   

    def __init__(
        self,
        estimator=None,
        n_estimators=50,
        sampling_size=0.1,
        args=None,
        bagging=False,
        split_test=False,
        max_samples=500,
        batch_test=15000
    ):
        self.sampling_size=sampling_size
        self.T=n_estimators
        self.estimator_=estimator
        self.args=args
        self.samplers=[]
        self.sampled_idxs=[]
        self.alphas=[]
        self.ws=[]
        self.test_probas=[]
        self.eps=1e-10
        self.bagging=bagging
        self.num_bags=10
        self.split_test=split_test
        self.max_samples=max_samples
        self.batch_test=batch_test
    def get_weak_learner(self,sampler_weights,X,y,X_test):
        
        num_trains=X.size(0)
        num_test=X_test.size(0)
        num_classes=int(y.max()+1)
        vec=np.arange(num_trains)
        sample_num=min(int(self.sampling_size*num_trains),self.max_samples)##sample_num should be less than max_samples
        if self.bagging:##this can be paralleled in future
            final_proba_all=0
            for bags in range(self.num_bags):
                idx=np.random.choice(vec,size=sample_num,replace=self.args.replacement, p=sampler_weights)
                while True:

                    if check_classnum(idx,y):
                        break
                    else:
                        idx=np.random.choice(vec,size=sample_num,replace=self.args.replacement, p=sampler_weights)
                estimator_=self.estimator_
                estimator_.fit(X[idx],y[idx])
                proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test)
                final_proba_all=proba_all+final_proba_all
            proba_all=final_proba_all/self.num_bags
        else:
            try:
                idx=np.random.choice(vec,size=sample_num,replace=self.args.replacement, p=sampler_weights)
            except:
                import ipdb
                ipdb.set_trace()


            if check_classnum(idx,y):
                new_class_idx=np.unique(y[idx])
                estimator_=self.estimator_
                estimator_.fit(X[idx],y[idx])
                proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test)
                
            else:
                ##choice 1: get label idx mapping from sampled dataset to full dataset
                not_sampledclass=np.setdiff1d(np.unique(y),np.unique(y[idx]))
                new_class_idx=np.unique(y[idx])
                
                # ### choice 2 :add not sampled class to train_dataset
                # not_sampledclass=np.setdiff1d(np.unique(y),np.unique(y[idx]))
                # for classnum in not_sampledclass:
                #     classidx=y==classnum
                #     vec_class=vec[classidx]
                #     sampler_weights_class=sampler_weights[classidx]
                #     sampler_weights_class=sampler_weights_class/np.sum(sampler_weights_class)
                #     idx_class=np.random.choice(vec_class,size=1,replace=self.args.replacement, p=sampler_weights_class)
                #     idx=np.append(idx,idx_class[0])
                # ############
                proba_all=np.zeros((num_trains+num_test,num_classes))
                if len(new_class_idx)==1 and not self.split_test:###for small dataset resample
                    resample=0
                    while True:
                        
                        idx=np.random.choice(vec,size=sample_num,replace=self.args.replacement, p=sampler_weights)
                        new_class_idx=np.unique(y[idx])
                        if len(new_class_idx)>1:
                            estimator_=self.estimator_
                            estimator_.fit(X[idx],y[idx])
                            
                            proba_all_predictor=estimator_.predict_proba(np.concatenate([X,X_test],axis=0))
                            for newidx in range(len(new_class_idx)):
                                proba_all[:,int(new_class_idx[newidx])]=proba_all_predictor[:,newidx]
                            break
                        else:
                            resample+=1
                            if resample>100:
                                
                                proba_all[:,int(new_class_idx[0])]=1
                                break
                    # import ipdb
                    # ipdb.set_trace()
                else:##using estimator
                    estimator_=self.estimator_
                    estimator_.fit(X[idx],y[idx])
                    
                    proba_all_predictor=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test)
                    for newidx in range(len(new_class_idx)):
                        proba_all[:,new_class_idx[newidx]]=proba_all_predictor[:,newidx]
                
                # predict=estimator_.predict(np.concatenate([X,X_test],axis=0))
                # import ipdb
                # ipdb.set_trace()
        return proba_all,idx
    def fit(self,X,y,X_test):
        ##
        num_trains=X.size(0)

        self.ws.append(np.ones(num_trains)/num_trains)
        self.F_prev=None
        for i in range(self.T):# boosting
            sampler_weights=self.ws[i]
            proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)

            test_proba=proba_all[num_trains:]
            # pred=np.argmax(proba_all[:num_trains],axis=1)
            train_proba=proba_all[:num_trains]
            try:
                y_pred=train_proba[np.arange(num_trains),y.int()]

            except:
                raise ValueError
            if self.F_prev is None :

                self.F_prev=y_pred
                gamma_t=1
            else:##boosting, find gamma_t
                F_prev=self.F_prev
                h=y_pred
                target=np.ones_like(h)
                gamma_t = update_model(target, F_prev, h)
                self.F_prev=F_prev + gamma_t * h
            
            self.alphas.append(gamma_t)##get alpha_t
            self.sampled_idxs.append(idx)##save sampled results
            self.test_probas.append(test_proba)##save test proba.
            next_sampler_weights=copy.deepcopy(sampler_weights)
            # next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
            # next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
            # # Z_norm=np.sum(next_sampler_weights)
            
            # next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            self.ws.append(next_sampler_weights)
    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)

    def predict_proba(self, X):

        all_probas=[self.alphas[i]*self.test_probas[i] for i in range(len(self.alphas))]
        # all_probas=Parallel(n_jobs=32)(delayed(lambda x : self.alphas[x]*self.test_probas[x])(i) for i in range(len(self.alphas)) )
        all_proba=sum(all_probas)
        # Reduce
        proba = all_proba / sum(self.alphas)
        # import ipdb
        # ipdb.set_trace()
        return proba

    def predict_log_proba(self, X):



        log_proba = np.log(self.predict_proba(X))

        return log_proba