# -*- coding: utf-8 -*-
# Author: Sumanta Mukherjee <sumanm03@in.ibm.com>
# License: Apache-2.0 License

import os
import pandas as pd
import torch
import random, argparse
from sklearn.preprocessing import MinMaxScaler
from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import *
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

# seeding
seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())


def get_dataset_name(filename: str):
    file_basename = os.path.basename(filename)
    return file_basename.split("_")[1]


def select_best_mode_by_dataset(filename: str, 
                                target_col: str = "file_name",
                                metric_col: str = "VUS-PR",
                                mode_col: str = "MODE",
                                greater_is_better: bool = True,
                                ):
    df = pd.read_csv(filename, sep=',', header='infer', index_col=None)
    performance = {}
    counter = {}
    for target_file, metric, mode in zip(df[target_col], df[metric_col], df[mode_col]):
        dset_name = get_dataset_name(target_file)
        if dset_name not in performance:
            performance[dset_name] = {}
            counter[dset_name] = {}
        
        if mode not in performance[dset_name]:
            performance[dset_name][mode] = 0
            counter[dset_name][mode] = 0
        
        performance[dset_name][mode] += float(metric)
        counter[dset_name][mode] += 1
    
    best_metric = {}
    for dset_name in performance:
        mode_names = []
        mode_performance = []
        for mode in performance[dset_name]:
            mode_names.append(mode)
            mode_performance.append(performance[dset_name][mode] / counter[dset_name][mode])
        if greater_is_better:
            index = np.argmax(mode_performance)
        else:
            index = np.argmin(mode_performance)
        best_metric[dset_name] = mode_names[index]
    return best_metric


if __name__ == '__main__':

    ## ArgumentParser
    parser = argparse.ArgumentParser(description='Running TSPulse')
    parser.add_argument('--filename', type=str, default='001_NAB_id_1_Facility_tr_1007_1st_2014.csv')
    parser.add_argument('--data_direc', type=str, default='Datasets/TSB-AD-U/')
    parser.add_argument('--save', type=bool, default=False)
    parser.add_argument('--AD_Name', type=str, default='TSPulse')
    
    # Optional argument to enable mode selection for TSPulse
    parser.add_argument('--tuning_results', 
                        type=str, 
                        default=None, 
                        help="Provide the resource file with tuning data experiment results. "
                             "Required by TSPulse algorithm for optimal data dependent mode selection.")
    parser.add_argument('--prediction_mode', 
                        type=str, 
                        default='#', 
                        help="Mode specification for TSPulse algorithm.") 
    args = parser.parse_args()

    df = pd.read_csv(args.data_direc + args.filename).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    label = df['Label'].astype(int).to_numpy()

    slidingWindow = find_length_rank(data, rank=1)
    train_index = args.filename.split('.')[0].split('_')[-3]
    data_train = data[:int(train_index), :]
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[args.AD_Name]
    
    if 'prediction_mode' in Optimal_Det_HP: 
        if args.prediction_mode != "#":
            Optimal_Det_HP['prediction_mode'] = args.prediction_mode
        elif (args.tuning_results is not None) and os.path.isfile(args.tuning_results):
            lookup = select_best_mode_by_dataset(args.tuning_results)
            dset_name = get_dataset_name(args.filename)
            Optimal_Det_HP['prediction_mode'] = lookup.get(dset_name, 'time')

    if args.AD_Name in Semisupervise_AD_Pool:
        output = run_Semisupervise_AD(args.AD_Name, data_train, data, **Optimal_Det_HP)
    elif args.AD_Name in Unsupervise_AD_Pool:
        output = run_Unsupervise_AD(args.AD_Name, data, **Optimal_Det_HP)
    else:
        raise Exception(f"{args.AD_Name} is not defined")

    if isinstance(output, np.ndarray):
        output = MinMaxScaler(feature_range=(0,1)).fit_transform(output.reshape(-1,1)).ravel()
        evaluation_result = get_metrics(output, label, slidingWindow=slidingWindow, pred=output > (np.mean(output)+3*np.std(output)))
        print('Evaluation Result: ', evaluation_result)
    else:
        print(f'At {args.filename}: '+output)