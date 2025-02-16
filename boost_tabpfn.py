import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from scripts import tabular_metrics
from sklearn.ensemble._gb_losses import LeastSquaresError, MultinomialDeviance
def check_classnum(idx,y):
    if len(np.unique(y[idx])) !=len(np.unique(y)) :
        return False
    else:
        return np.all(np.equal(np.unique(y[idx]),np.unique(y)))
from utils import splitting_predict_proba
class SamplingAdaboost():
   

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
        new=False
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
        self.new=new
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
                proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.new)
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
                proba_all=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.new)
                
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
                            
                            proba_all_predictor=estimator_.predict_proba(np.concatenate([X,X_test],axis=0),return_logits=self.new)
                            for newidx in range(len(new_class_idx)):
                                proba_all[:,int(new_class_idx[newidx])]=proba_all_predictor[:,newidx]
                            break
                        else:
                            resample+=1
                            if resample>0:
                                
                                proba_all[:,int(new_class_idx[0])]=1
                                break
                    # import ipdb
                    # ipdb.set_trace()
                else:##using estimator
                    estimator_=self.estimator_
                    estimator_.fit(X[idx],y[idx])
                    
                    proba_all_predictor=splitting_predict_proba(estimator_,np.concatenate([X,X_test],axis=0),self.batch_test,return_logits=self.new)
                    for newidx in range(len(new_class_idx)):
                        proba_all[:,new_class_idx[newidx]]=proba_all_predictor[:,newidx]
                
                # predict=estimator_.predict(np.concatenate([X,X_test],axis=0))
                # import ipdb
                # ipdb.set_trace()
        return proba_all,idx
    def fit(self,X,y,X_test):
        ##
        num_trains=X.size(0)
        num_test=X_test.size(0)
        num_classes=int(y.max()+1)
        vec=np.arange(num_trains)
        self.ws.append(np.ones(num_trains)/num_trains)
        for i in range(self.T):# boosting
            sampler_weights=self.ws[i]
            proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)

            test_proba=proba_all[num_trains:]
            pred=np.argmax(proba_all[:num_trains],axis=1)
            true_idx=torch.Tensor(pred)==torch.Tensor(y)
            false_idx=~true_idx
#
            # eps_t=sum(sampler_weights[false_idx])
            eps_t=sum(sampler_weights[false_idx])
            resample=0
            while 1-eps_t<1/num_classes:### need weak learner to be better than random guess
                proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)

                test_proba=proba_all[num_trains:]
                
                pred=np.argmax(proba_all[:num_trains],axis=1)
                true_idx=torch.Tensor(pred)==torch.Tensor(y)
                false_idx=~true_idx
    #
                eps_t=sum(sampler_weights[false_idx])
                resample+=1
                if resample>0:
                    break

##############################

            # alpha_t=np.log((1-eps_t)/eps_t)* 1/num_classes##version 2 adaboost
            alpha_t=np.log(((1-eps_t)+self.eps)/(eps_t+self.eps)) + np.log(num_classes-1)##weaklearner version boost
            if np.isnan(alpha_t):
                import ipdb
                ipdb.set_trace()
            self.alphas.append(alpha_t)##get alpha_t
            self.sampled_idxs.append(idx)##save sampled results
            self.test_probas.append(test_proba)##save test proba.
            next_sampler_weights=copy.deepcopy(sampler_weights)
            next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
            next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
            # Z_norm=np.sum(next_sampler_weights)
            
            next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            self.ws.append(next_sampler_weights)

    def fit_debug(self,X,y,X_test, y_test):
        num_trains=X.size(0)
        num_classes=int(y.max()+1)
        if self.loss=="CE":
            self.loss_=MultinomialDeviance(num_classes)
        elif self.loss=="MSE":
            self.loss_=LeastSquaresError()
        elif self.loss=="EXP":
            raise NotImplementedError
        else:
            raise KeyError
        self.F_prev = []
        self.ws.append(np.ones(num_trains)/num_trains)
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
            proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)
            test_proba=proba_all[num_trains:]
            train_proba=proba_all[:num_trains]
            pred=np.argmax(proba_all[:num_trains],axis=1)
            true_idx=torch.Tensor(pred)==torch.Tensor(y)
            false_idx=~true_idx
            eps_t=sum(sampler_weights[false_idx])
            resample=0
            while 1-eps_t<1/num_classes:
                proba_all,idx=self.get_weak_learner(sampler_weights,X,y,X_test)
                test_proba=proba_all[num_trains:]
                pred=np.argmax(proba_all[:num_trains],axis=1)
                true_idx=torch.Tensor(pred)==torch.Tensor(y)
                false_idx=~true_idx
                eps_t=sum(sampler_weights[false_idx])
                resample+=1
                if resample>100:
                    break
            alpha_t=np.log(((1-eps_t)+self.eps)/(eps_t+self.eps)) + np.log(num_classes-1)
            if np.isnan(alpha_t):
                import ipdb
                ipdb.set_trace()
            self.alphas.append(alpha_t)
            self.sampled_idxs.append(idx)
            self.test_probas.append(test_proba)
            next_sampler_weights=copy.deepcopy(sampler_weights)
            next_sampler_weights[true_idx]=sampler_weights[true_idx]*np.exp(-alpha_t)
            next_sampler_weights[false_idx]=sampler_weights[false_idx]*np.exp(alpha_t)
            next_sampler_weights=next_sampler_weights/sum(next_sampler_weights)
            self.ws.append(next_sampler_weights)

    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)

    def predict_proba(self, X):

        all_probas=[self.alphas[i]*self.test_probas[i] for i in range(len(self.alphas))]
        # all_probas=Parallel(n_jobs=32)(delayed(lambda x : self.alphas[x]*self.test_probas[x])(i) for i in range(len(self.alphas)) )
        all_proba=np.array(sum(all_probas))
        # Reduce
        # import ipdb
        # ipdb.set_trace()
        if self.new:
            proba =torch.softmax(torch.Tensor(all_proba),dim=-1).numpy()
        else:
            
            proba=np.divide(all_proba, np.sum(all_proba,axis=-1)[:, np.newaxis]+self.eps)
        # import ipdb
        # ipdb.set_trace()
        return proba
    
    def predict_proba_debug(self, X,y=None):
        
        all_probas=[self.alphas[i]*self.test_probas[i] for i in range(len(self.alphas))]
        # all_probas=Parallel(n_jobs=32)(delayed(lambda x : self.alphas[x]*self.test_probas[x])(i) for i in range(len(self.alphas)) )
        all_proba=np.array(sum(all_probas))
        if self.debug:
            for i in range(len(self.test_F_prev)):
                test_proba_proba =torch.softmax(torch.Tensor(self.test_F_prev[i]),dim=-1).numpy()
                self.test_auc.append(tabular_metrics.auc_metric(y,test_proba_proba))
        # Reduce
        # import ipdb
        # ipdb.set_trace()
        if self.new:
            proba =torch.softmax(torch.Tensor(all_proba),dim=-1).numpy()
        else:
            proba=np.divide(all_proba, np.sum(all_proba,axis=-1)[:, np.newaxis]+self.eps)
        # import ipdb
        # ipdb.set_trace()
        return proba, self.boost_residuals, self.boost_loss, self.train_auc, self.test_auc, self.test_F_prev, self.test_boost_loss, self.test_boost_residuals

    def predict_log_proba(self, X):



        log_proba = np.log(self.predict_proba(X))

        return log_proba