import pandas as pd
import argparse
from enum import Enum

class ModelName(Enum):
    Tabpfn = 'Tabpfn(one)'
    Bagging_without_replacement = 'bagging_tabpfn(repalcement =False)'
    Bagging_with_replacement = 'bagging(replacement=True)'
    adaboost_pfn = 'Adaboost_tabpfn'
    GboostV2_exp = 'Gboost_V2(exphadamard)'
    GboostV2_ada = 'Gboost_V2(adaboost)'
    Autogluon = 'Autogluon'

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--filename', type=str,default="results/results.xlsx", help='the filename of the csv file')
    args = parser.parse_args()
    return args

def get_modelname_by_index(indexes):
    names = [member.name for member in ModelName]
    models = [member.value for member in ModelName]
    try:
        return [names[index] for index in indexes], [models[index] for index in indexes]
    except IndexError:
        raise

def calculate_mean_rank(df, names):
    fail_list = []
    mean_rank = {name: [] for name in names}
    for _, row in df.iterrows():
        try:
            roc = [float(x) for x in row[1:].tolist()]
            roc_name = list(zip(names, roc))
            roc_name = sorted(roc_name, key=lambda x: x[-1], reverse=True)
            for i, (name, _) in enumerate(roc_name): 
                mean_rank[name].append(i+1)
        except:
            fail_list.append(row['name'])
    return fail_list, mean_rank

def calculate_ranks(numbers):
    # Sort the numbers in descending order while maintaining original indices
    sorted_numbers = sorted(enumerate(numbers), key=lambda x: x[1], reverse=True)

    ranks = [0] * len(numbers)  # Initialize ranks
    current_rank = 1  # Start with rank 1

    for i, (original_index, num) in enumerate(sorted_numbers):
        # Check if it's not the first element and the number is the same as the previous
        if i > 0 and num == sorted_numbers[i - 1][1]:
            ranks[original_index] = ranks[sorted_numbers[i - 1][0]]
        else:
            ranks[original_index] = current_rank
            current_rank += 1

    return ranks

def main():
    args = parse_args()
    df = pd.read_excel(args.filename)
    rankss=[]
    for _, row in df.iterrows():

        rocs = [float(x) for x in row[1:-1].tolist()]
        ranks=calculate_ranks(rocs)
        rankss.append(ranks)
    with open("ranksforsmall.txt","w") as f:
        for ranks in rankss:
            f.write(str(ranks))
            f.write("\n")
    return rankss

main()