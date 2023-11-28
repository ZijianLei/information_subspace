import os

# TASK = sys.argv[1]
# MODEL = sys.argv[2]
# METHOD = sys.argv[2] ,'ib_adapter','mnli'
TASK_DICT = ['rte','mrpc','stsb','cola','sst2','qnli','qqp','mnli','mnli_mm']
TASK_DICT =['rte','mrpc','stsb','sst2','qnli','qqp']
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
WITH_IB = ['True','False']
Train_Sample= [100,200,400]
SEED = [1111,2222,3333]
def rename_all_folders(directory):
    for folder_name in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, folder_name)):
            new_folder_name = folder_name + ''
            os.rename(os.path.join(directory, folder_name), os.path.join(directory, new_folder_name))

# Specify the directory you want to start from
# start_dir = '/path/to/your/directory'
# rename_all_folders(start_dir)
for seed in SEED:
    for with_ib in WITH_IB:
        for train_sample in Train_Sample:
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
                        epoch = 20
                    else:
                        bs = 8
                        epoch = 5
                        epoch = 20
                    if TASK in SPECIAL_METRICS:
                        METRIC = SPECIAL_METRICS[TASK]
                    else:
                        METRIC = "accuracy"
                    for lr in LR_DICT:
                        for dropout_rate in DROP_RATE:
                            for selection_step in  SELECTION_STEP:
                                if METHOD == 'ib_adapter':
                                    id_n='100000'   
                                    start_dir = f"/home/datasets/zjlei/fewshot-glue-result/{METHOD}/{TASK}/{TASK}-{bs}-{epoch}-{lr}-{id_n}-{selection_step}-{dropout_rate}-random-{seed}-withib_{with_ib}-train_sample-{train_sample}-beta-1e-5"
                                    new_name = start_dir+'-ib_dim-384'
                                    if os.path.isdir(start_dir):
                                        os.rename(start_dir, new_name)