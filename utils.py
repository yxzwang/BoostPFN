import numpy as np
def splitting_predict_proba(classifier,x_test,test_batch=1000,return_logits=False,bagging=False):
    if bagging:
        y_pred=[]
        test_num=x_test.shape[0]
        for x_idx in range(0,test_num,test_batch):
            y_pred.append(classifier.predict_proba(x_test[x_idx:min((x_idx+test_batch),test_num)]))
        prediction_proba=np.concatenate(y_pred,axis=0)
    else:
        y_pred=[]
        test_num=x_test.shape[0]
        for x_idx in range(0,test_num,test_batch):
            y_pred.append(classifier.predict_proba(x_test[x_idx:min((x_idx+test_batch),test_num)],return_logits=return_logits))
        prediction_proba=np.concatenate(y_pred,axis=0)
    
    return prediction_proba