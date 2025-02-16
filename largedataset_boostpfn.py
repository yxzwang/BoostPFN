#srun --cpus-per-task=4 --mem-per-cpu=4G --gres=gpu:1 --job-name=hadamard python largedataset_searchfulldatasets.py --modelname gboost_tabpfnV2 --gpu 0 --sampling_size 0.001 --test_batch 50000 --seed 5 --maxsample 500 --updating hadamard --endnum 0 --step 10 --startnum -1 --ensemble_num 1000
import time
import torch
import numpy as np
import os
import time
from scripts.model_builder import get_default_spec, save_model, load_model_only_inference
from scripts.transformer_prediction_interface import transformer_predict, get_params_from_config, TabPFNClassifier
from scripts.differentiable_pfn_evaluation import eval_model, eval_model_range
from sklearn.ensemble import BaggingClassifier
from datasets import load_openml_list, open_cc_dids, open_cc_valid_dids, test_dids_classification
import torch
from scripts import tabular_metrics
import random
import argparse
from boost_tabpfn import SamplingAdaboost
from utils import splitting_predict_proba
import logging
logger = logging.getLogger(__file__)
def get_root_logger(file=True):
    format = "%(asctime)-10s %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(format=format, datefmt=datefmt)
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)

    if file:
        handler = logging.FileHandler("log.txt")
        format = logging.Formatter(format, datefmt)
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger
def get_args():
    parser = argparse.ArgumentParser(description="None")
    parser.add_argument("--gpu", default=0,type=int,help="gpu index")
    parser.add_argument("--seed", default=0,type=int,help="seed")
    parser.add_argument("--modelname", default="tabpfn",type=str,help="modelname")#choice=["tabpfn","bagging_tabpfn","adaboost_tabfpn"]
    parser.add_argument("--ensemble_num", default=10,type=int,help="ensemble")
    parser.add_argument("--sampling_size", default=0.1,type=float,help="sampling size")
    parser.add_argument("--replacement",action="store_true",default=False, help="whether replacement when sampling. False =不放回")
    parser.add_argument("--test_batch", default=50000,type=int,help="batchsize for testing")
    parser.add_argument("--batch_size", default=512,type=int,help="batchsize for minibatch training")
    parser.add_argument("--train_portion", default=0.01,type=float,help="training portion")
    parser.add_argument("--maxsample", default=500,type=int,help="maxsample for training")
    parser.add_argument("--loss", default="CE",type=str,help="loss for gboost")#loss for gboost## MSE,CE
    parser.add_argument("--updating", default="exphadamard",type=str,help="weight updating for gboost ")#updating for gboost ##adaboost,hadamard,exphadamard
    parser.add_argument("--wl_num", default=2,type=int,help="number of weaklearners sampled in each round, only used for V4")
    parser.add_argument("--debug",default=False, action = 'store_true')
    parser.add_argument("--startnum", default=0,type=int,help="start num from aindex, use 10 datasets")
    parser.add_argument("--step", default=10,type=int,help="how many datasets run in a row")
    parser.add_argument("--endnum", default=0,type=int,help="start num from aindex, use 10 datasets")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    logger=get_root_logger()
    baselinemodels=["LightGBM","autogluon","XGBoost"]
    args=get_args()
    base_path = '.'
    max_samples = 10000000
    #####from high instances to low

    final_idxs=[125,133,136,138,147,155,156,157,158,161,162,266,1177,1179,1240,1502,1597,40515,40520,40672,41960,42206,42397,42746,45654,45665,45669]

    if args.startnum<0:
        ###最终选定的数据集，通过endnum和step的方法选择
        if args.step ==0:

            run_indexes=final_idxs
        else:
            if args.endnum+args.step>len(final_idxs):

        
                run_indexes=final_idxs[args.endnum:]
            else:
                run_indexes=final_idxs[args.endnum:args.endnum+args.step]
    else:
        raise KeyError

   
    # good_indexes=[]
    # bad_indexes=[]
    # for ind in run_indexes:
    #     try:
    #         cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(args,[ind], multiclass=True, shuffled=True, filter_for_nan=False, max_samples = max_samples, num_feats=100, return_capped=True)
    #         good_indexes.append(ind)
    #     except:
    #         bad_indexes.append(ind)
    #         logger.warning(f"###############bad index {ind}###################")
    #         continue
    
    cc_test_datasets_multiclass, cc_test_datasets_multiclass_df = load_openml_list(args,run_indexes, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = max_samples, num_feats=100, return_capped=True)

    # Loading longer OpenML Datasets for generalization experiments (optional)
    # test_datasets_multiclass, test_datasets_multiclass_df = load_openml_list(test_dids_classification, multiclass=True, shuffled=True, filter_for_nan=False, max_samples = 10000, num_feats=100, return_capped=True)




    model_string, longer, task_type = '', 1, 'multiclass'
    eval_positions = [1000]

    N_en=1
    test_datasets= cc_test_datasets_multiclass

    # [(i, test_datasets[i][0]) for i in range(len(test_datasets))]
    if args.gpu>-1:
        device=f"cuda:{args.gpu}"
    else:
        device="cpu"
    def set_seed(seed: int):
   
        random.seed(seed)
        np.random.seed(seed)



        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    set_seed(args.seed)
    ensemblenum=args.ensemble_num
    datasetnames=[]
    shapes=[]
    rocs=[]
    ces=[]
    accs=[]
    times=[]
    modelname=args.modelname
    datasetportion=args.sampling_size
    # modelname="tabpfn"
    fail_datasets=[]
    logger.warning(f"#################################running model {modelname}#################################")
    for i in range(len(test_datasets)):
        datasetportion=args.sampling_size
        evaluation_dataset_index = i # Index of the dataset to predict
        datasetidx=run_indexes[i]
        ds = test_datasets[evaluation_dataset_index]
        logger.warning(f'Evaluation dataset name: {ds[0]} shape {ds[1].shape}')

        

        xs, ys = ds[1].clone(), ds[2].clone()
        eval_position = xs.shape[0] // 2
        train_xs, train_ys = xs[0:eval_position], ys[0:eval_position]
        test_xs, test_ys = xs[eval_position:], ys[eval_position:]
        starttime=time.time()
        try:
            if modelname=="tabpfn":
                import math
                vec=np.arange(train_xs.size(0))
                idx=np.array([])
                num_trains=train_xs.size(0)
                for classnum in np.unique(train_ys):
                    classidx=train_ys==classnum
                    vec_class=vec[classidx]
                    num_classsample=vec_class.shape[0]
                    samplesize=math.ceil(args.maxsample/num_trains*num_classsample)
                    samplesize=min(samplesize,num_classsample)
                    sampler_weights_class=np.ones(num_classsample)/num_classsample
                    idx_class=np.random.choice(vec_class,size=samplesize,replace=args.replacement, p=sampler_weights_class)
                    idx=np.append(idx,idx_class)
                train_xs=train_xs[idx]
                train_ys=train_ys[idx]
                classifier = TabPFNClassifier(device=device,N_ensemble_configurations=N_en)
                classifier.fit(train_xs, train_ys,overwrite_warning=True)
                prediction_ =splitting_predict_proba(classifier,test_xs,test_batch=args.test_batch) 
            elif modelname=="tabpfn_onetime":

                classifier = TabPFNClassifier(device=device,N_ensemble_configurations=N_en)
                classifier.fit(train_xs, train_ys)
                prediction_ =splitting_predict_proba(classifier,test_xs,test_batch=args.test_batch) 
            elif modelname=="bagging_tabpfn":
                datasetportion=args.sampling_size
                maxsample=args.maxsample
                datasetportion=maxsample/train_xs.shape[0]

                try:
                    classifier=BaggingClassifier(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),n_estimators=ensemblenum,max_samples=datasetportion,bootstrap=args.replacement)
                    # import ipdb
                    # ipdb.set_trace()
                    classifier.fit(train_xs, train_ys)
                    prediction_ = splitting_predict_proba(classifier,test_xs,test_batch=args.test_batch,bagging=True) 
                except:
                    from baggingself import SamplingBagging
                    classifier=SamplingBagging(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),n_estimators=ensemblenum,max_samples=maxsample,batch_test=args.test_batch,return_logits=False, args=args)
                    # import ipdb
                    # ipdb.set_trace()
                    classifier.fit(train_xs, train_ys,test_xs)
                    prediction_ = classifier.predict_proba(test_xs)
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
            elif modelname=="adaboost_tabpfn":
                datasetportion=args.sampling_size
                classifier=SamplingAdaboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="newadaboost_tabpfn":
                datasetportion=args.sampling_size
                classifier=SamplingAdaboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch,new=True)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="adaboost_bagging_tabpfn":
                datasetportion=args.sampling_size
                classifier=SamplingAdaboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,bagging=True)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = splitting_predict_proba(classifier,test_xs,test_batch=args.test_batch) 
            elif modelname=="minibatch_tabpfn":
                from minibatch_tabpfn import MiniBatch
                datasetportion=args.sampling_size
                classifier=MiniBatch(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),args,args.batch_size,train_portion=args.train_portion,split_test=True,batch_test=args.test_batch)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="adaboostnoweight_tabpfn":
                from boost_tabpfn_noweight import SamplingAdaboost_noweight
                datasetportion=args.sampling_size
                classifier=SamplingAdaboost_noweight(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="gboostnoweight_tabpfn":
                from gradient_boost_tabpfn_noweight import SamplingGradientboost_noweight
                datasetportion=args.sampling_size
                classifier=SamplingGradientboost_noweight(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="gboost_tabpfn":
                from gradient_boost_tabpfn import SamplingGradientboost
                datasetportion=args.sampling_size
                classifier=SamplingGradientboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="gboost_tabpfnV2":
                from gradient_boost_tabpfn import SamplingGradientboost
                datasetportion=args.sampling_size
                classifier=SamplingGradientboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch,version=2)
                classifier.fit(train_xs, train_ys,test_xs)
                
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="gboost_tabpfnV3":
                from gradient_boost_tabpfn import SamplingGradientboost
                datasetportion=args.sampling_size
                classifier=SamplingGradientboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch,version=3)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="gboost_tabpfnV4":
                from gradient_boost_tabpfn import SamplingGradientboost
                datasetportion=args.sampling_size
                classifier=SamplingGradientboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch,version=4)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            elif modelname=="gboost_tabpfnV5":
                from gradient_boost_tabpfn import SamplingGradientboost
                datasetportion=args.sampling_size
                classifier=SamplingGradientboost(TabPFNClassifier(device=device,N_ensemble_configurations=N_en),ensemblenum,args.sampling_size,args,split_test=True,
            max_samples=args.maxsample,
            batch_test=args.test_batch,version=5)
                classifier.fit(train_xs, train_ys,test_xs)
                prediction_ = classifier.predict_proba(test_xs)
            else:
                
                raise KeyError
            
        except:
            logger.warning(f"########################prediction failure on {ds[0]}################################")
            if args.debug:
                raise
            fail_datasets.append(ds[0])
            datasetnames.append(ds[0])
            shapes.append(ds[1].shape)
            rocs.append(0.0)
            ces.append(0.0)
            accs.append(0.0)
            times.append(0.0)
            continue
        roc= tabular_metrics.auc_metric(test_ys, prediction_)
        acc=tabular_metrics.accuracy_metric(test_ys, prediction_)

        prediction_[prediction_>1]=1
        prediction_[prediction_<0]=0
        ce=tabular_metrics.cross_entropy(test_ys, prediction_)
        
        roc=float(roc)
        ce=float(ce)

        endtime=time.time()
        # 'AUC', float(roc), 'Cross Entropy', float(ce)
        
        logger.warning(f"roc: {roc} CE: {ce}")
        logger.warning(f"consuming time : {endtime-starttime}")
        datasetnames.append(ds[0])
        shapes.append(ds[1].shape)
        rocs.append(roc)
        ces.append(ce)
        accs.append(acc)
        times.append(endtime-starttime)
    datasetportion=args.sampling_size
    if modelname=="tabpfn":
        savedir=f"largeresults/{min(run_indexes)}_{max(run_indexes)}_{args.modelname}_{N_en}_{args.maxsample}"
        resultname=f"{savedir}/{args.seed}.txt"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    elif modelname=="minibatch_tabpfn":
        resultname=f"largeresults/{args.modelname}_{args.batch_size}x{args.train_portion}/{args.seed}.txt"
        savedir=f"largeresults/{args.modelname}_{args.batch_size}x{args.train_portion}"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    
    elif modelname.startswith("gboost_tabpfn"):
        savedir=f"largeresults/{min(run_indexes)}_{max(run_indexes)}_{args.modelname}_{N_en}_{datasetportion}x{ensemblenum}_{args.loss}x{args.updating}_{args.maxsample}"
        resultname=f"{savedir}/{args.seed}.txt"
        
        if "4" in modelname:
            savedir=f"largeresults/{args.modelname}_{datasetportion}x{ensemblenum}_{args.loss}x{args.updating}_{args.maxsample}_wl={args.wl_num}"
            resultname=savedir+f"/{args.seed}.txt"
        if not os.path.exists(savedir):
            os.mkdir(savedir)
    else:
        if args.replacement==True:
            savedir=f"largeresults/{min(run_indexes)}_{max(run_indexes)}_{args.modelname}_{N_en}_{datasetportion}x{ensemblenum}_{args.maxsample}_replacement={args.replacement}"
        else:
            savedir=f"largeresults/{min(run_indexes)}_{max(run_indexes)}_{args.modelname}_{N_en}__{datasetportion}x{ensemblenum}_{args.maxsample}"
        resultname=f"{savedir}/{args.seed}.txt"
        
        if not os.path.exists(savedir):
            os.mkdir(savedir)

    with open(resultname,"w") as f:
        f.write("|   |")
        for datasetname in datasetnames:
            f.write(f" {datasetname} |")
        f.write("\n")
        f.write("| -------- |")
        for datasetname in datasetnames:
            
               
            f.write(" -------- |")
        f.write("\n")
        if modelname=="tabpfn":
            f.write(f"|{modelname}|")
        elif modelname=="minibatch_tabpfn":
            f.write(f"|{modelname},{args.batch_size},{args.train_portion}|")
        else:
            f.write(f"|{modelname},{datasetportion}x{ensemblenum},{args.maxsample}|")
        for roc in rocs:

            f.write(" {:.4} |".format(roc))
        f.write("\n")
        f.write(f"|{modelname},ce|")
        for ce in ces:

            f.write(" {:.4} |".format(ce))
        f.write("\n")
        f.write(f"|{modelname},acc|")
        for acc in accs:

            f.write(" {:.4} |".format(acc))
        f.write("\n")
        f.write(f"|{modelname},time(s)|")
        for t in times:

            f.write(" {:.4} |".format(t))
    logger.warning(f"saved results for {modelname}")
