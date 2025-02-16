import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import time
import numpy as np
from scipy.optimize import minimize
from joblib import Parallel,delayed
from scripts import tabular_metrics
from sklearn.ensemble._gb_losses import LeastSquaresError,BinomialDeviance,ExponentialLoss,MultinomialDeviance
from scipy.special import  logsumexp

# def loss_function(y, y_pred):
#     """
#     定义损失函数，这里假设是平方损失
#     """
#     return np.sum((y - y_pred) ** 2)
# def CEloss_function(y, y_pred):##
#     """
#     定义损失函数，这里假设是平方损失
#     """

#     return torch.nn.functional.nll_loss(torch.softmax(y_pred,dim=-1),y,reduction="sum")
# def objective_function(gamma, y, F_prev, h):
#     """
#     定义优化目标函数
#     """
#     y_pred = F_prev + gamma * h
#     return CEloss_function(y, y_pred)



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
class SamplingGradientboost():
   

    def __init__(
        self,
        estimator=None,
        n_estimators=50,
        sampling_size=0.1,
        args=None,
        bagging=False,
        split_test=False,
        max_samples=500,
        batch_test=15000,
        return_logits=True,
        version=1,
    ):
        self.sampling_size=sampling_size
        self.T=n_estimators
        self.estimator_=estimator
        self.args=args
        if args is None:
            self.replacement=False
        else:
            self.replacement=args.replacement
        self.samplers=[]
        self.sampled_idxs=[]
        self.alphas=[]
        self.ws=[]
        self.test_probas=[]
        self.eps=np.finfo(np.float32).eps
        self.bagging=bagging
        self.num_bags=10
        self.split_test=split_test
        self.max_samples=max_samples
        self.batch_test=batch_test
        self.loss=args.loss
        self.return_logits=return_logits
        self.updating=args.updating
        self.version=version
        self.wl_num=args.wl_num
        self.debug=args.debug
        if self.debug:
            self.boost_residuals=[]
            self.boost_loss=[]
            self.test_boost_loss=[]
            self.test_boost_residuals = []
            self.train_auc=[]
            self.test_auc=[]
            self.test_F_prev=[]
    def get_weak_learner(self,sampler_weights,X,y,X_test):
        
        num_trains=X.size(0)
        num_test=X_test.size(0)
        num_classes=int(y.max()+1)
        vec=np.arange(num_trains)
        if self.split_test:
            sample_num=self.max_samples
        else:
            sample_num=min(int(self.sampling_size*num_trains),self.max_samples)##sample_num should be less than max_samples
        if self.bagging:##this can be paralleled in future
            # final_proba_all=0
            # for bags in range(self.num_bags):
            #     idx=np.random.choice(vec,size=sample_num,replace=self.replacement, p=sampler_weights)
            #     while True:

            #         if check_classnum(idx,y):
            #             break
            #         else:
            #             idx=np.random.choice(vec,size=sample_num,replace=self.replacement, p=sampler_weights)
            #     estimator_=self.estimator_
            #     estimator_.fit(X[idx],y[idx])
            #     proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.return_logits)
            #     final_proba_all=proba_all+final_proba_all
            # proba_all=final_proba_all/self.num_bags
            pass
        else:
            try:
                idx=np.random.choice(vec,size=sample_num,replace=self.replacement, p=sampler_weights)
            except:
                import ipdb
                ipdb.set_trace()


            if check_classnum(idx,y):
                estimator_=self.estimator_
                estimator_.fit(X[idx],y[idx],overwrite_warning=True)
                proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.return_logits)
                
            elif self.version==5:
                print("adding ")
                ### choice 2 :add not sampled class to train_dataset
                not_sampledclass=np.setdiff1d(np.unique(y),np.unique(y[idx]))
                for classnum in not_sampledclass:
                    classidx=y==classnum
                    vec_class=vec[classidx]
                    sampler_weights_class=sampler_weights[classidx]
                    sampler_weights_class=sampler_weights_class/np.sum(sampler_weights_class)
                    idx_class=np.random.choice(vec_class,size=1,replace=self.replacement, p=sampler_weights_class)
                    idx=np.append(idx,idx_class[0])
                estimator_=self.estimator_
                estimator_.fit(X[idx],y[idx],overwrite_warning=True)
                proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.return_logits)
            else:
                new_class_idx=np.unique(y[idx])

                # ############
                proba_all=np.zeros((num_trains+num_test,num_classes))
                if len(new_class_idx)==1 and not self.split_test:###for small dataset resample ###this is aimed for the condition that only sample one class
                    resample=0
                    while True:
                        
                        idx=np.random.choice(vec,size=sample_num,replace=self.replacement, p=sampler_weights)
                        new_class_idx=np.unique(y[idx])
                        if len(new_class_idx)>1:
                            estimator_=self.estimator_
                            estimator_.fit(X[idx],y[idx],overwrite_warning=True)
                            
                            proba_all_predictor=estimator_.predict_proba(np.concatenate([X,X_test],axis=0),return_logits=self.return_logits)
                            for newidx in range(len(new_class_idx)):
                                proba_all[:,int(new_class_idx[newidx])]=proba_all_predictor[:,newidx]
                            break
                        else:
                            resample+=1
                            if resample>100:
                                if self.return_logits:
                                    proba_all[:,int(new_class_idx[0])]=99
                                else:
                                    proba_all[:,int(new_class_idx[0])]=1
                                break
                    # import ipdb
                    # ipdb.set_trace()
                else:##using estimator
                    estimator_=self.estimator_
                    estimator_.fit(X[idx],y[idx],overwrite_warning=True)
                    
                    proba_all_predictor=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.return_logits)
                    for newidx in range(len(new_class_idx)):
                        proba_all[:,new_class_idx[newidx]]=proba_all_predictor[:,newidx]###give zero probability for not sampled class
                
                # predict=estimator_.predict(np.concatenate([X,X_test],axis=0))
                # import ipdb
                # ipdb.set_trace()
        return proba_all,idx
    def get_weak_learners(self,ws_list,X,y,X_test,residuals):
        ###get weaklearners and compare to find the best one.
        num_trains=X.size(0)
        bestdistance=9999
        new_sampler_weights=ws_list[-1]
        best_proba_all,best_idx=None,None
        for i in range(self.wl_num):
            proba_all,idx=self.get_weak_learner(new_sampler_weights,X,y,X_test)
            train_proba=proba_all[:num_trains]
            distance=np.linalg.norm(train_proba-residuals)
            if distance<bestdistance:
                bestdistance=distance
                best_proba_all,best_idx=proba_all,idx

        return best_proba_all,best_idx
    def get_3weak_learners(self,X,y,X_test,residuals):
        ###get weaklearners and compare to find the best one.
        num_trains=X.size(0)
        bestdistance=9999
        past_sampler_weights=self.ws[-2]
        new_sampler_weights=self.ws[-1]
        uniformweights=np.ones(num_trains)/num_trains
        best_proba_all,best_idx=None,None
        learners=[uniformweights,past_sampler_weights,new_sampler_weights]
        
        for i in range(len(learners)):
            idxweights=learners[i]
            proba_all,idx=self.get_weak_learner(idxweights,X,y,X_test)
            train_proba=proba_all[:num_trains]
            distance=np.linalg.norm(train_proba-residuals)
            if distance<bestdistance:
                bestdistance=distance
                best_proba_all,best_idx=proba_all,idx
                best_weights=idxweights
        ##rewrite this round's weights
        self.ws.pop()
        self.ws.append(best_weights)
        return best_proba_all,best_idx
    def get_init_raw_predictions(self,y_pred):
        eps = np.finfo(np.float32).eps
        probas = np.clip(y_pred, eps, 1 - eps)
        raw_predictions = np.log(probas).astype(np.float64)
        return raw_predictions
    def update_model(self,y, F_prev, h):
        """
        更新模型参数
        """
        result = minimize(
            lambda gamma: self.loss_(y,F_prev+gamma*h),
            x0=0.0,  # 初始猜测值
            method='L-BFGS-B'  # 选择适当的优化方法
        )
        gamma_optimal = result.x[0]
        return gamma_optimal
    def fit(self,X,y,X_test):
        ##

        num_trains=X.size(0)
        num_classes=int(y.max()+1)
        if self.loss=="CE":
            self.loss_=MultinomialDeviance(num_classes)
            # if num_classes>2:
            #     self.loss_=MultinomialDeviance(num_classes)
            # else:
            #     self.loss_=BinomialDeviance(num_classes)
        elif self.loss=="MSE":
            self.loss_=LeastSquaresError()
        elif self.loss=="EXP":
            raise NotImplementedError
        else:
            raise KeyError
        self.ws.append(np.ones(num_trains)/num_trains)
        self.F_prev=[]
        if self.debug:##initial residuals
            residuals=[]
            for k in range(num_classes):
                target_y = np.array(y == k, dtype=np.float64)
                
                residuals.append(self.loss_.negative_gradient(target_y,np.zeros([num_trains,num_classes]),k))
            residuals=np.stack(residuals,axis=-1)
            # if num_classes>2:
            #     residuals=[]
            #     for k in range(num_classes):
            #         target_y = np.array(y == k, dtype=np.float64)
                    
            #         residuals.append(self.loss_.negative_gradient(target_y,np.zeros([num_trains,num_classes]),k))
            #     residuals=np.stack(residuals,axis=-1)
            # else:
            #     residuals=self.loss_.negative_gradient(y,np.zeros([num_trains,num_classes]))
            
            self.boost_residuals.append(residuals)
            pass
        for i in range(self.T):# boosting
            sampler_weights=self.ws[i]
            if self.version==4:
                if i==0:

                    proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test,)##don't need to compare at first round
                else:
                    proba_all,idx=self.get_weak_learners(self.ws,X,y,X_test,self.residuals)
            elif self.version==7:
                if i==0:

                    proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test,)##don't need to compare at first round
                else:
                    proba_all,idx=self.get_3weak_learners(X,y,X_test,self.residuals)
            else:

                proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)

            test_proba=proba_all[num_trains:]
            train_proba=proba_all[:num_trains]
            try:
                y_pred=train_proba
            except:
                raise ValueError
            if len(self.F_prev)==0:
                F_pred=np.zeros_like(y_pred)
            else:##boosting, find gamma_t
                F_pred=self.F_prev[-1]
            h=y_pred
            try:
                gamma_t=self.update_model(y, F_pred, h)
            except:
                if self.args.debug:
                    raise
                pass
                # import ipdb
                # ipdb.set_trace()
            self.F_prev.append(F_pred + gamma_t * h)

            if self.debug:
                
                train_proba_proba =torch.softmax(torch.Tensor(self.F_prev[-1]),dim=-1).numpy()
                self.train_auc.append(tabular_metrics.auc_metric(y,train_proba_proba))
                self.boost_loss.append(self.loss_(y,F_pred + gamma_t * h))
                ###get testresults
                if len(self.test_F_prev)==0:
                    test_F_pred=np.zeros_like(test_proba)
                else:##boosting, find gamma_t
                    test_F_pred=self.test_F_prev[-1]
                self.test_F_prev.append(test_F_pred + gamma_t * test_proba)
                
            self.alphas.append(gamma_t)##get alpha_t
            self.sampled_idxs.append(idx)##save sampled results
            self.test_probas.append(test_proba)##save test proba.
            next_sampler_weights=copy.deepcopy(sampler_weights)
            residuals=[]
            if self.version==1:
                for k in range(num_classes):
                    target_y = np.array(y == k, dtype=np.float64)

                    residuals.append(self.loss_.negative_gradient(target_y,F_pred,k))

            else:
                for k in range(num_classes):
                    target_y = np.array(y == k, dtype=np.float64)
                    residuals.append(self.loss_.negative_gradient(target_y,F_pred + gamma_t * h,k))
            residuals=np.stack(residuals,axis=-1)
            # if num_classes>2:
            #     if self.version==1:
            #         for k in range(num_classes):
            #             target_y = np.array(y == k, dtype=np.float64)

            #             residuals.append(self.loss_.negative_gradient(target_y,F_pred,k))

            #     else:
            #         for k in range(num_classes):
            #             target_y = np.array(y == k, dtype=np.float64)
            #             residuals.append(self.loss_.negative_gradient(target_y,F_pred + gamma_t * h,k))
            #     residuals=np.stack(residuals,axis=-1)
            # else:
            #     residuals=self.loss_.negative_gradient(target_y,F_pred + gamma_t * h)
            self.residuals=residuals###save residual for next round
            
            # import ipdb
            # ipdb.set_trace()

            if self.version==3:
                target_residuals=np.linalg.norm(residuals,axis=-1)
            else:
                target_residuals=residuals[range(num_trains),y.int()]

            if self.debug:
                self.boost_residuals.append((residuals,target_residuals))
            ### updating weights based on residuals
            if self.updating=="adaboost":
            ###adaboost updating
            #adding updating weights
                pred=np.argmax(proba_all[:num_trains],axis=1)
                true_idx=torch.Tensor(pred)==torch.Tensor(y)
                false_idx=~true_idx
    #
                # eps_t=sum(sampler_weights[false_idx])
                eps_t=sum(sampler_weights[false_idx])
                alpha_t=np.log(((1-eps_t)+self.eps)/(eps_t+self.eps)) + np.log(num_classes-1)
                next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
                next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
            # # Z_norm=np.sum(next_sampler_weights)
            #######residual updating
            
            ##residual for CE loss
            elif self.updating=="hadamard":
                if self.version==6:
                    pred=np.argmax(proba_all[:num_trains],axis=1)
                    true_idx=torch.Tensor(pred)==torch.Tensor(y)
                    false_idx=~true_idx
        #
                    # eps_t=sum(sampler_weights[false_idx])
                    # eps_t=np.sum(target_residuals)
                    alpha_t=np.sum(np.abs(target_residuals))
                    next_sampler_weights[true_idx]=sampler_weights[true_idx]*(-alpha_t)
                    next_sampler_weights[false_idx]=sampler_weights[false_idx]*(alpha_t)
            # # Z_norm=np.sum(next_sampler_weights)
                else:
                    next_sampler_weights=next_sampler_weights*(target_residuals)+self.eps
            elif self.updating=="exphadamard":
                if self.version==6:
                    pred=np.argmax(proba_all[:num_trains],axis=1)
                    true_idx=torch.Tensor(pred)==torch.Tensor(y)
                    false_idx=~true_idx
        #
                    # eps_t=sum(sampler_weights[false_idx])
                    # eps_t=np.sum(target_residuals)
                    alpha_t=np.sum(np.abs(target_residuals))
                    next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
                    next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
                else:
                    next_sampler_weights=next_sampler_weights*np.exp(target_residuals)
                # if np.isinf(next_sampler_weights).any():
                #     import ipdb
                #     ipdb.set_trace()

                # next_sampler_weights[np.isinf(next_sampler_weights)]=1/self.eps
            elif self.updating=="loghadamard":
                next_sampler_weights=next_sampler_weights*np.log(target_residuals)+self.eps
            elif self.updating=="none":
                pass
            else:
                raise KeyError
            if np.isinf(next_sampler_weights).any():
                import ipdb
                ipdb.set_trace()
            
            next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            eps = self.eps
            next_sampler_weights = np.clip(next_sampler_weights, eps, 1 - eps)
            next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            if np.isnan(next_sampler_weights).any():
                import ipdb
                ipdb.set_trace()
            self.ws.append(next_sampler_weights.astype(np.float64))
    def fit_debug(self,X,y,X_test,y_test):
        ##

        num_trains=X.size(0)
        num_classes=int(y.max()+1)
        if self.loss=="CE":
            self.loss_=MultinomialDeviance(num_classes)
            # if num_classes>2:
            #     self.loss_=MultinomialDeviance(num_classes)
            # else:
            #     self.loss_=BinomialDeviance(num_classes)
        elif self.loss=="MSE":
            self.loss_=LeastSquaresError()
        elif self.loss=="EXP":
            raise NotImplementedError
        else:
            raise KeyError
        self.ws.append(np.ones(num_trains)/num_trains)
        self.F_prev=[]
        if self.debug:##initial residuals
            residuals=[]
            for k in range(num_classes):
                target_y = np.array(y == k, dtype=np.float64)
                
                residuals.append(self.loss_.negative_gradient(target_y,np.zeros([num_trains,num_classes]),k))
            residuals=np.stack(residuals,axis=-1)
            
            self.boost_residuals.append(residuals)

            residuals=[]
            for k in range(num_classes):
                target_y = np.array(y == k, dtype=np.float64)
                
                residuals.append(self.loss_.negative_gradient(target_y,np.zeros([num_trains,num_classes]),k))
            residuals=np.stack(residuals,axis=-1)
            
            self.test_boost_residuals.append(residuals)
            pass
        for i in range(self.T):# boosting
            sampler_weights=self.ws[i]
            if self.version==4:
                if i==0:

                    proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test,)##don't need to compare at first round
                else:
                    proba_all,idx=self.get_weak_learners(self.ws,X,y,X_test,self.residuals)
            elif self.version==7:
                if i==0:

                    proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test,)##don't need to compare at first round
                else:
                    proba_all,idx=self.get_3weak_learners(X,y,X_test,self.residuals)
            else:

                proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)

            test_proba=proba_all[num_trains:]
            train_proba=proba_all[:num_trains]
            try:
                y_pred=train_proba
            except:
                raise ValueError
            if len(self.F_prev)==0:
                F_pred=np.zeros_like(y_pred)
            else:##boosting, find gamma_t
                F_pred=self.F_prev[-1]
            h=y_pred
            try:
                gamma_t=self.update_model(y, F_pred, h)
            except:
                if self.args.debug:
                    raise
                pass
                # import ipdb
                # ipdb.set_trace()
            self.F_prev.append(F_pred + gamma_t * h)

            if self.debug:
                
                train_proba_proba =torch.softmax(torch.Tensor(self.F_prev[-1]),dim=-1).numpy()
                self.train_auc.append(tabular_metrics.auc_metric(y,train_proba_proba))
                self.boost_loss.append(self.loss_(y,F_pred + gamma_t * h))
                ###get testresults
                if len(self.test_F_prev)==0:
                    test_F_pred=np.zeros_like(test_proba)
                else:##boosting, find gamma_t
                    test_F_pred=self.test_F_prev[-1]
                self.test_F_prev.append(test_F_pred + gamma_t * test_proba)
                self.test_boost_loss.append(self.loss_(y_test,test_F_pred + gamma_t * test_proba))
                
            self.alphas.append(gamma_t)##get alpha_t
            self.sampled_idxs.append(idx)##save sampled results
            self.test_probas.append(test_proba)##save test proba.
            next_sampler_weights=copy.deepcopy(sampler_weights)
            residuals=[]
            if self.version==1:
                for k in range(num_classes):
                    target_y = np.array(y == k, dtype=np.float64)

                    residuals.append(self.loss_.negative_gradient(target_y,F_pred,k))

            else:
                for k in range(num_classes):
                    target_y = np.array(y == k, dtype=np.float64)
                    residuals.append(self.loss_.negative_gradient(target_y,F_pred + gamma_t * h,k))
            residuals=np.stack(residuals,axis=-1)
            # if num_classes>2:
            #     if self.version==1:
            #         for k in range(num_classes):
            #             target_y = np.array(y == k, dtype=np.float64)

            #             residuals.append(self.loss_.negative_gradient(target_y,F_pred,k))

            #     else:
            #         for k in range(num_classes):
            #             target_y = np.array(y == k, dtype=np.float64)
            #             residuals.append(self.loss_.negative_gradient(target_y,F_pred + gamma_t * h,k))
            #     residuals=np.stack(residuals,axis=-1)
            # else:
            #     residuals=self.loss_.negative_gradient(target_y,F_pred + gamma_t * h)
            self.residuals=residuals###save residual for next round
            
            # import ipdb
            # ipdb.set_trace()

            if self.version==3:
                target_residuals=np.linalg.norm(residuals,axis=-1)
            else:
                target_residuals=residuals[range(num_trains),y.int()]

            if self.debug:
                self.boost_residuals.append((residuals,target_residuals))
            ### updating weights based on residuals
            if self.updating=="adaboost":
            ###adaboost updating
            #adding updating weights
                pred=np.argmax(proba_all[:num_trains],axis=1)
                true_idx=torch.Tensor(pred)==torch.Tensor(y)
                false_idx=~true_idx
    #
                # eps_t=sum(sampler_weights[false_idx])
                eps_t=sum(sampler_weights[false_idx])
                alpha_t=np.log(((1-eps_t)+self.eps)/(eps_t+self.eps)) + np.log(num_classes-1)
                next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
                next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
            # # Z_norm=np.sum(next_sampler_weights)
            #######residual updating
            
            ##residual for CE loss
            elif self.updating=="hadamard":
                if self.version==6:
                    pred=np.argmax(proba_all[:num_trains],axis=1)
                    true_idx=torch.Tensor(pred)==torch.Tensor(y)
                    false_idx=~true_idx
        #
                    # eps_t=sum(sampler_weights[false_idx])
                    # eps_t=np.sum(target_residuals)
                    alpha_t=np.sum(np.abs(target_residuals))
                    next_sampler_weights[true_idx]=sampler_weights[true_idx]*(-alpha_t)
                    next_sampler_weights[false_idx]=sampler_weights[false_idx]*(alpha_t)
            # # Z_norm=np.sum(next_sampler_weights)
                else:
                    next_sampler_weights=next_sampler_weights*(target_residuals)+self.eps
            elif self.updating=="exphadamard":
                if self.version==6:
                    pred=np.argmax(proba_all[:num_trains],axis=1)
                    true_idx=torch.Tensor(pred)==torch.Tensor(y)
                    false_idx=~true_idx
        #
                    # eps_t=sum(sampler_weights[false_idx])
                    # eps_t=np.sum(target_residuals)
                    alpha_t=np.sum(np.abs(target_residuals))
                    next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
                    next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
                else:
                    next_sampler_weights=next_sampler_weights*np.exp(target_residuals)
                # if np.isinf(next_sampler_weights).any():
                #     import ipdb
                #     ipdb.set_trace()

                # next_sampler_weights[np.isinf(next_sampler_weights)]=1/self.eps
            elif self.updating=="loghadamard":
                next_sampler_weights=next_sampler_weights*np.log(target_residuals)+self.eps
            elif self.updating=="none":
                pass
            else:
                raise KeyError
            if np.isinf(next_sampler_weights).any():
                import ipdb
                ipdb.set_trace()
            
            next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            eps = self.eps
            next_sampler_weights = np.clip(next_sampler_weights, eps, 1 - eps)
            next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            if np.isnan(next_sampler_weights).any():
                import ipdb
                ipdb.set_trace()
            self.ws.append(next_sampler_weights.astype(np.float64))
    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)

    def predict_proba(self, X):
        
        all_probas=[self.alphas[i]*self.test_probas[i] for i in range(len(self.alphas))]
        # all_probas=Parallel(n_jobs=32)(delayed(lambda x : self.alphas[x]*self.test_probas[x])(i) for i in range(len(self.alphas)) )
        all_proba=torch.Tensor(np.array(sum(all_probas)))
        # Reduce
        # import ipdb
        # ipdb.set_trace()
        proba =torch.softmax(all_proba,dim=-1).numpy()
        # import ipdb
        # ipdb.set_trace()
        return proba
    def predict_proba_debug(self, X,y=None):
        
        all_probas=[self.alphas[i]*self.test_probas[i] for i in range(len(self.alphas))]
        # all_probas=Parallel(n_jobs=32)(delayed(lambda x : self.alphas[x]*self.test_probas[x])(i) for i in range(len(self.alphas)) )
        all_proba=torch.Tensor(np.array(sum(all_probas)))
        if self.debug:
            for i in range(len(self.test_F_prev)):
                test_proba_proba =torch.softmax(torch.Tensor(self.test_F_prev[i]),dim=-1).numpy()
                self.test_auc.append(tabular_metrics.auc_metric(y,test_proba_proba))
        # Reduce
        # import ipdb
        # ipdb.set_trace()
        proba =torch.softmax(all_proba,dim=-1).numpy()
        # import ipdb
        # ipdb.set_trace()
        return proba, self.boost_residuals, self.boost_loss, self.train_auc, self.test_auc, self.test_F_prev, self.test_boost_loss, self.test_boost_residuals

    def predict_log_proba(self, X):



        log_proba = np.log(self.predict_proba(X))

        return log_proba