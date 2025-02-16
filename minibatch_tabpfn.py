import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
import time
from joblib import Parallel,delayed
import random
import math
def check_classnum(idx,y):
    if len(np.unique(y[idx])) !=len(np.unique(y)) :
        return False
    else:
        return np.all(np.equal(np.unique(y[idx]),np.unique(y)))
from utils import splitting_predict_proba

class MiniBatch():
   

    def __init__(
        self,
        estimator=None,
        args=None,
        batch_size=512,
        split_test=False,
        train_portion=0.1,
        batch_test=15000
    ):
        self.estimator_=estimator
        self.args=args
        self.eps=1e-10
        self.batch_size=batch_size
        self.split_test=split_test
        self.batch_test=batch_test

        self.train_portion=train_portion
    # def get_mini_batch(self,X,y):
        
    #     return idxs#idxs for each minibatch
    def fit(self,X,y,X_test):
        ##
        self.test_probas=[]
        
        num_trains=X.size(0)
        num_classes=int(y.max()+1)
        train_num=int(self.train_portion*num_trains)
        class_idxs=[]###class_num*batch_num*batch_size
        train_batchnum=train_num//self.batch_size
        # vec=np.arange(num_trains)
        # train_idx=np.random.choice(vec,size=train_num,replace=self.args.replacement,p=np.ones(num_trains)/num_trains)
        # X_train=X[train_idx]
        # y_train=y[train_idx]
        for classidx in range(num_classes):
            i_classidx_all=np.where(y == classidx)[0]
            try:
                i_classidx=np.random.choice(i_classidx_all,size=max(math.ceil(len(i_classidx_all)*self.train_portion),train_batchnum),replace=self.args.replacement)
            except:
                neednum=max(math.ceil(len(i_classidx_all)*self.train_portion),train_batchnum)
                totalnum=len(i_classidx_all)
                repeatnum=neednum//totalnum
                residualnum=neednum%totalnum
                i_classidx=np.repeat(i_classidx_all,repeatnum,axis=0)
                if residualnum>0:
                    i_classidx=np.append(i_classidx,np.random.choice(i_classidx_all,size=residualnum,replace=self.args.replacement))

            np.random.shuffle(i_classidx)
            
            classsamplenum=len(i_classidx)
            class_step=classsamplenum//train_batchnum
            if class_step<1:
                import ipdb
                ipdb.set_trace()

            
            class_idx=Parallel(n_jobs=32)(delayed(lambda x : i_classidx[x:min(x+class_step,classsamplenum)])(startidx) for startidx in range(0,classsamplenum,class_step) )
            class_idxs.append(class_idx)

            # for startidx in range(0,classsamplenum,class_step):
            #     endidx=min(startidx+class_step,classsamplenum)
                
            #     class_idxb=i_classidx[startidx:endidx]
            #     class_idxs[classidx].append(class_idxb)

        for batchidx in range(train_batchnum):

            idx=np.concatenate([class_idxs[classidx][batchidx] for classidx in range(num_classes)],axis=0)
        
            self.estimator_.fit(X[idx], y[idx])
            if self.split_test:
                prediction_=splitting_predict_proba(self.estimator_,X_test,self.batch_test)
            else:
                prediction_ = self.estimator_.predict_proba(X_test)
            self.test_probas.append(prediction_)
    def predict(self, X):
        predicted_probabilitiy = self.predict_proba(X)
        return self.classes_.take((np.argmax(predicted_probabilitiy, axis=1)), axis=0)

    def predict_proba(self, X):

        all_probas=self.test_probas
        # all_probas=Parallel(n_jobs=32)(delayed(lambda x : self.test_probas[x])(i) for i in range(len(self.test_probas)) )
        all_proba=sum(all_probas)
        # Reduce
        proba = all_proba / len(self.test_probas)
        # import ipdb
        # ipdb.set_trace()
        return proba

    def predict_log_proba(self, X):



        log_proba = np.log(self.predict_proba(X))

        return log_proba