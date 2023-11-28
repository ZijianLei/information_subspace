import json
import os
import sys
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
# from tasks.utils import *

# TASK = sys.argv[1]
# MODEL = sys.argv[2]
# METHOD = sys.argv[2] ,'ib_adapter','mnli'
TASK_DICT = ['rte','mrpc','stsb','cola','sst2','qnli','qqp']
TASK_DICT = ['rte']

# LR_DICT = ['1e-5','5e-5','1e-4','5e-4','1e-3']
LR_DICT = ['5e-3','1e-3','1e-4','5e-4','1e-5','5e-5']
LR_DICT = ['1e-3']

small_set = ['rte','mrpc','stsb','cola',]

SPECIAL_METRICS = {
    'cb' : 'f1',
    'mrpc' : 'combined_score',

    'cola' : 'matthews_correlation',
    'stsb' : 'combined_score'
    
}
METHOD_DICT  =['ib_adapter']
dropout_dict = ['0.7','0.9']
step_dict = [20,40,60,80]
# dropout_dict = [0.68,0.7,0.72,0.74,0.76,0.8]
# METHOD_DICT = ['finetune','FISH','sequencial_adapter','parallel_adapter','lora','said','ib_adapter']
# epoch = 25
epoch = 10
for METHOD in METHOD_DICT:
    avg = []
    std = []
    s = ""
    # print(METHOD)
    for TASK in TASK_DICT:
        for drop in dropout_dict:
            mean_dict = []
            # std_dict = []
            best_mean = 0
            best_std = 0
            best_lr = 0
            if TASK in small_set:
                bs = 8
            else:
                bs = 32
            
            if TASK in SPECIAL_METRICS:
                METRIC = SPECIAL_METRICS[TASK]
            else:
                METRIC = "accuracy"
            for step in step_dict:
                lr = '1e-3'
                if METHOD == 'ib_adapter':
                    id_n='100002'
                    # id_n = '32768'
                    # lr='1e-2'
                    # lr = '*'
                    # files = glob(f"/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{lr}-{id_n}-0.9-ipt-*-incremental/eval_results.json")
                    files = glob(f"/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{epoch}-{lr}-{id_n}-{step}-{drop}-ipt-*-lid/eval_results.json")
                    print(files)
                best_score = []
                for f in files:
                    metrics = json.load(open(f, 'r'))
                    # print(metrics["eval_"+METRIC],f)
                    # if metrics["eval_"+METRIC] > best_score:
                    best_score.append(metrics["eval_"+METRIC]) 
                    # print(best_score)
                mean_score = np.mean(best_score)
                # print(mean_score)
                mean_dict.append(mean_score)
            print(step_dict,mean_dict)
            if TASK == 'rte':
                color='red'
            else:
                color = 'blue'
            if drop == '0.7':
                linestyle = '-'
            else:
                linestyle = '--'
            plt.plot(step_dict,mean_dict,linestyle,color=color,label = 'task {:s} dropout rate {:s}'.format(TASK,drop))
    plt.legend()
    plt.xlabel("step_size")
    plt.ylabel("acc")
    plt.savefig("rate")