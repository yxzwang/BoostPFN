# Introduction
Code for AISTATS2025 paper: Prior-Fitted Networks Scale to Larger Datasets When Treated as Weak Learners. 

Initial code for submission. Further revision coming soon.
# Requirements
See ```requirements.txt``` which is from previous work of [TabPFN v1](https://github.com/PriorLabs/TabPFN/tree/v1.0.0).
# Reproduction of  BoostPFN in Table 3

To run small datasets lower than 5000:

```python main_10times.py --modelname gboost_tabpfnV2 --updating exphadamard ```

To run large datasets:

```python largedataset_boostpfn.py --modelname gboost_tabpfnV2 --gpu 0 --sampling_size 0.001 --test_batch 50000 --seed 5 --maxsample 500 --updating exphadamard --endnum 0 --step 100 --startnum -1 --ensemble_num 10```

Main Arguments:

```--updating```: the updating method for boostpfn, can choose from exphadamard,hadamard,adaboost

```--ensemble_num```: the number of boosting rounds


