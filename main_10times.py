import time
import torch
import numpy as np
import os

from scripts.model_builder import get_default_spec, save_model, load_model_only_inference
from scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, TabPFNClassifier
from scripts.differentiable_pfn_evaluation import eval_model, eval_model_range
from sklearn.ensemble import BaggingClassifier,GradientBoostingClassifier
from boost_tabpfn_noweight import SamplingAdaboost_noweight
from datasets import load_openml_list, open_cc_dids, open_cc_valid_dids, test_dids_classification
import torch
from scripts import tabular_metrics
import random
import argparse
from boost_tabpfn import SamplingAdaboost
def get_args():
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("--gpu", default=0,type=int,help="gpu index")
    parser.add_argument("--seed", default=0,type=int,help="seed")
    parser.add_argument("--modelname", default="tabpfn",type=str,help="modelname")#choice=["tabpfn","bagging_tabpfn","adaboost_tabfpn"]
    parser.add_argument("--ensemble_num", default=10,type=int,help="ensemble")
    parser.add_argument("--sampling_size", default=0.5,type=float,help="sampling size")
    parser.add_argument("--replacement",action="store_true",default=False, help="whether replacement when sampling. False =不放回")
    parser.add_argument("--batch_size", default=64,type=int,help="batchsize for minibatch training")
    parser.add_argument("--train_portion", default=1,type=float,help="training portion")
    parser.add_argument("--loss", default="CE",type=str,help="loss for gboost")#loss for gboost## MSE,CE
    parser.add_argument("--updating", default="hadamard",type=str,help="weight updating for gboost ")#updating for gboost ##adaboost,hadamard,exphadamard
    parser.add_argument("--wl_num", default=2,type=int,help="number of weaklearners sampled in each round, only used for V4")
    parser.add_argument("--debug",default=False, action = 'store_true')
    args = parser.parse_args()
    return args
def run_onetime(test_datasets,args,run=0):
    ensemblenum=args.ensemble_num
    datasetnames=[]
    shapes=[]
    rocs=[]
    ces=[]
    accs=[]
    modelname=args.modelname
    datasetportion=args.sampling_size
    if args.debug:
        debugresults="debug/smallresults.txt"
    print(f"#################################running model {modelname} at  run {run}#################################")
    for i in range(len(test_datasets)):
        evaluation_dataset_index = i # Index of the dataset to predict
        ds = test_datasets[evaluation_dataset_index]
        print(f'Evaluation dataset name: {ds[0]} shape {ds[1].shape}')

        datasetnames.append(ds[0])
        shapes.append(ds[1].shape)

        xs, ys = ds[1].clone(), ds[2].clone()
        eval_position = xs.shape[0] // 2
        train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
        test_xs, test_ys = xs[eval_position:], ys[eval_position:]

        if modelname=="tabpfn":
            classifier = TabPFNClassifier(device=device,N_ensemble_configurations=32)
            classifier.fit(train_xs, train_ys)
            prediction_ = classifier.predict_proba(test_xs)
        elif modelname=="bagging_tabpfn":
            while True:
                try:
                    datasetportion=args.sampling_size
                    classifier=BaggingClassifier(TabPFNClassifier(device=device,N_ensemble_configurations=32),n_estimators=ensemblenum,max_samples=datasetportion,bootstrap=args.replacement)
                    classifier.fit(train_xs, train_ys)
                    prediction_ = classifier.predict_proba(test_xs)
                    break
                except:
                    print("######## re-bagging ######")

            # try:
            #     datasetportion=0.15
            #     classifier=BaggingClassifier(TabPFNClassifier(device=device,),n_estimators=ensemblenum,max_samples=datasetportion)
            #     classifier.fit(train_xs, train_ys)
            #     prediction_ = classifier.predict_proba(test_xs)
            # except:
            #     datasetportion=0.2
            #     classifier=BaggingClassifier(TabPFNClassifier(device=device,),n_estimators=ensemblenum,max_samples=datasetportion)
            #     classifier.fit(train_xs, train_ys)
            #     prediction_ = classifier.predict_proba(test_xs)
        
        
        elif modelname=="gboost_tabpfnV2":
            from gradient_boost_tabpfn import SamplingGradientboost
            datasetportion=args.sampling_size
            classifier=SamplingGradientboost(TabPFNClassifier(device=device,N_ensemble_configurations=32),ensemblenum,args.sampling_size,args,version=2)
            classifier.fit(train_xs, train_ys,test_xs)
            prediction_ = classifier.predict_proba(test_xs)
            if args.debug:
                prediction_,boost_residuals,boost_loss,train_auc,test_auc,test_F_prev = classifier.predict_proba_debug(test_xs,test_ys)
                debugresults="debug/smallresults"
        
        else:
            raise KeyError
        roc= tabular_metrics.auc_metric(test_ys, prediction_)
        try:
            ce=tabular_metrics.cross_entropy(test_ys, prediction_)
        except:
            prediction_[prediction_>1]=1
            prediction_[prediction_<0]=0
            ce=tabular_metrics.cross_entropy(test_ys, prediction_)
        
        acc=tabular_metrics.accuracy_metric(test_ys, prediction_)
        # roc, ce = tabular_metrics.auc_metric(test_ys, prediction_), 0
        roc=float(roc)
        ce=float(ce)
        # 'AUC', float(roc), 'Cross Entropy', float(ce)
        
        print("roc:",roc,"  CE:",ce,"  ACC:",acc)
        rocs.append(roc)
        ces.append(ce)
        accs.append(acc)
    return datasetnames,rocs,ces,accs
if __name__=="__main__":
    args=get_args()
    base_path = '.'
    max_samples = 10000
    bptt = 10000

    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(args,open_cc_dids, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = max_samples, num_feats=100, return_capped=True)

    # Loading longer OpenML Datasets for generalization experiments (optional)
    # test_datasets_multiclass, test_datasets_multiclass_df = load_openml_list(test_dids_classification, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True)

    random.seed(0)

    # def get_datasets(selector, task_type, suite='cc'):
    #     if task_type == 'binary':
    #         ds = valid_datasets_binary if selector == 'valid' else test_datasets_binary
    #     else:
    #         if suite == 'openml':
    #             ds = valid_datasets_multiclass if selector == 'valid' else test_datasets_multiclass
    #         elif suite == 'cc':
    #             ds = cc_valid_datasets_multiclass if selector == 'valid' else cc_test_datasets_multiclass
    #         else:
    #             raise Exception("Unknown suite")
    #     return ds
    def get_datasets(selector, task_type, suite='cc'):
            

        return cc_test_datasets_multiclass
    
    model_string, longer, task_type = '', 1, 'multiclass'
    eval_positions = [1000]
    bptt = 2000
        
    test_datasets, valid_datasets = get_datasets('test', task_type, suite='cc'), get_datasets('valid', task_type, suite='cc')
    # [(i, test_datasets[i][0]) for i in range(len(test_datasets))]
    if args.gpu>-1:
        device=f"cuda:{args.gpu}"
    else:
        device="cpu"
    ########set random seeds
    def set_seed(seed: int):
   
        random.seed(seed)
        np.random.seed(seed)



        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(args.seed)
    # modelname="tabpfn"
    rocss=[]
    cess=[]
    accss=[]
    if args.debug:
        datasetnames,rocs,ces,accs=run_onetime(test_datasets,args)
    else:
        for run in range(10):
            datasetnames,rocs,ces,accs=run_onetime(test_datasets,args,run)
            rocss.append(rocs)
            cess.append(ces)
            accss.append(accs)
        def postprocess(results):
            results=np.array(results)
            results_avg=np.mean(results,axis=0)## avg for different datasets. a list
            results_std=np.std(results,axis=0)
            avg_results=np.mean(results,axis=-1)
            totalavg=np.mean(avg_results)####avg for all datasets, a value
            totalstd=np.std(avg_results)
            return results_avg,results_std,totalavg,totalstd
        results=[postprocess(rocss),postprocess(cess),postprocess(accss)]### roc,ce,acc
        metricnames=["rocauc","ce","acc"]
        # roc_avg,roc_std,roc_totalavg,roc_total_std=postprocess(rocss)
        # ce_avg,ce_std,ce_totalavg,ce_total_std=postprocess(cess)
        # acc_avg,acc_std,acc_totalavg,acc_total_std=postprocess(accss)
        modelname=args.modelname
        ensemblenum=args.ensemble_num
        datasetportion=args.sampling_size
        if modelname=="tabpfn":
            resultname="results/tabpfn_baseline_10times.txt"
        elif modelname=="bagging_tabpfn":
            resultname=f"results/baggingp_tabpfn_{datasetportion}x{ensemblenum}_10times.txt"
        elif modelname=="minibatch_tabpfn":
            resultname=f"results/{args.modelname}_{args.batch_size}x{args.train_portion}_10times.txt"
        elif modelname.startswith("gboost_tabpfn"):
            resultname=f"results/{args.modelname}_{datasetportion}x{ensemblenum}_{args.loss}x{args.updating}_10times.txt"
            if "4" in modelname:
                resultname=f"results/{args.modelname}_{datasetportion}x{ensemblenum}_{args.loss}x{args.updating}_wl={args.wl_num}_10times.txt"
        else:
            resultname=f"results/{args.modelname}_{datasetportion}x{ensemblenum}_10times.txt"
        with open(resultname,"w") as f:
            f.write("|   |")
            for datasetname in datasetnames:

                f.write(f" {datasetname} |")
            f.write(f" Total Avg |")

            f.write("\n")
            f.write("| -------- |")
            for datasetname in datasetnames:
                f.write(" -------- |")
            f.write("| -------- |")

            f.write("\n")
            # if modelname=="tabpfn":
            #     f.write(f"|{modelname}|")
            # elif modelname=="minibatch_tabpfn":
            #     f.write(f"|{modelname},{args.batch_size},{args.train_portion}|")
            # else:
            #     f.write(f"|{modelname},{datasetportion}x{ensemblenum}|")
            
            for i in range(len(results)):##roc,ce,acc
                result=results[i]
                metricname=metricnames[i]
                if modelname=="tabpfn":
                    f.write(f"|{modelname},{metricname}|")
                elif modelname=="minibatch_tabpfn":
                    f.write(f"|{modelname},{args.batch_size},{args.train_portion},{metricname}|")
                else:
                    f.write(f"|{modelname},{datasetportion}x{ensemblenum},{metricname}|")
                
                for i in range(len(result[0])):
                    avg=result[0][i]
                
                    f.write(" {:.4} |".format(avg))
                f.write(" {:.4} |".format(result[2]))
                f.write("\n")

                if modelname=="tabpfn":
                    f.write(f"|std-{metricname}|")
                elif modelname=="minibatch_tabpfn":
                    f.write(f"|std-{metricname}|")
                else:
                    f.write(f"|std-{metricname}|")
                for i in range(len(result[0])):
                    std=result[1][i]
                
                    f.write("{:.4}|".format(std))
                f.write(" {:.4}|".format(result[3]))
                f.write("\n")

        print(f"saved results for {modelname}")
