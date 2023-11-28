import json
import os
import sys
import numpy as np
from glob import glob

# from tasks.utils import *

# TASK = sys.argv[1]
# MODEL = sys.argv[2]
# METHOD = sys.argv[2] ,'ib_adapter','mnli'
TASK_DICT = ['rte','mrpc','stsb','cola','sst2','qnli','qqp','mnli','mnli_mm']
TASK_DICT =['rte','mrpc','stsb','cola','sst2','qnli','qqp']
# TASK_DICT = ['rte','mrpc','stsb','sst2']
# LR_DICT = ['1e-5','5e-5','1e-4','5e-4','1e-3']
# LR_DICT = ['1e-3','1e-4','5e-4','1e-5','1e-6']
LR_DICT = ['5e-5','1e-5']
small_set = ['rte','mrpc','stsb','cola',]
DROP_RATE = ['1']
variantion_type = ['ipt','random']
SPECIAL_METRICS = {
    'cb' : 'f1',
    'mrpc' : 'combined_score',
    'mnli_mm' : 'accuracy_mm',
    'cola' : 'matthews_correlation',
    'stsb' : 'combined_score'
    
}
SELECTION_STEP = [100]
METHOD_DICT  =['sequencial_adapter','parallel_adapter']
METHOD_DICT = ['ib_adapter']
# METHOD_DICT = ['finetune','FISH','sequencial_adapter','parallel_adapter','lora','said','ib_adapter']
for v_type in variantion_type:
    for METHOD in METHOD_DICT:
        avg = []
        std = []
        s = ""
        # print(METHOD)
        for TASK in TASK_DICT:
            # print(TASK)
            best_mean = 0
            best_mean_mm = 0
            best_std = 0
            best_lr = 0
            if TASK in small_set:
                bs = 8
                # epoch = 25
                epoch = 10
            else:
                bs = 32
                epoch = 5
                epoch = 3
            if TASK in SPECIAL_METRICS:
                METRIC = SPECIAL_METRICS[TASK]
            else:
                METRIC = "accuracy"
            for lr in LR_DICT:
                for dropout_rate in DROP_RATE:
                    for selection_step in  SELECTION_STEP:
                        if METHOD == 'ib_adapter':
                            id_n='100002'
                            # id_n = '32768'
                            # lr='1e-2'
                            # lr = '*'
                            files = glob(f"/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{epoch}-{lr}-{id_n}-{selection_step}-{dropout_rate}-{v_type}-*-lid/eval_results.json")
                            # this file is finetune on full data result
                            files = glob(f"/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{epoch}-{lr}-{id_n}-{selection_step}-{dropout_rate}-{v_type}-*-dataib/eval_results.json")
                            # glob files
                            # files = 
                            # files = glob(f"/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{epoch}-{lr}-{id_n}-{selection_step}-{dropout_rate}-ipt-*-lid")
                            # for file in files:
                            #     new_file_name = file.replace(f"-{bs}-{lr}-",f"-{bs}-{epoch}-{lr}-")
                            #     os.rename(file,new_file_name)
                            #     # os.rename(f"/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{lr}-{id_n}-{selection_step}-{dropout_rate}-ipt-*-lid",f'/home/datasets/zjlei/glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{epoch}-{lr}-{id_n}-{selection_step}-{dropout_rate}-ipt-*-lid')
                            # print(files)


                        best_score = []
                        # best_score_mm = []
                        for f in files:
                            metrics = json.load(open(f, 'r'))
                            # print(metrics["eval_"+METRIC],f)
                            # if metrics["eval_"+METRIC] > best_score:
                            best_score.append(metrics["eval_"+METRIC])

                        temp_mean = np.mean(best_score)
                    
                        print( "drop_rate {:s} selection step {:.2f}".format(dropout_rate,selection_step),np.mean(best_score),np.std(best_score),lr)      
                        if temp_mean >= best_mean:
                            best_mean = temp_mean
                            best_std = np.std(best_score)
                            best_lr = lr
                
            avg.append(best_mean)
            # print(best_score)
            std.append(best_std)
            # print(best_lr)    
            # print(f"best_{METRIC}: {best_mean}, std: {best_std}, learning rate {best_lr}")
                # print(f"best_metrics: {best_metrics}")
                # print(f"best_file: {best_file_name}")
        # print(METHOD,np.mean(avg))

        avg = [100*i for i in avg]
        std = [100*i for i in std]
    
        
        for i in range(len(avg)):
            s += "& ${:.2f}_".format(avg[i])+"{"+"\pm{:.2f} ".format(std[i])+"}$"
        print(METHOD+v_type+s+"& {:.2f}\\".format(np.mean(avg))+'\\'+' \hline')
