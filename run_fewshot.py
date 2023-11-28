#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import torch.nn as nn
import datasets
import torch
import numpy as np
from datasets import load_dataset, load_metric
# from my_model.trajectory_sampling import *
import transformers

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    AdapterTrainer,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from Arguments import get_args
# import  matplotlib.pyplot as plt
import torch.utils.cpp_extension
import os
# import seaborn as sns
from task.glue.glue_datasets import glue_data
import shutil
# detect the DAAI running enviroment
envs = os.environ['CONDA_DEFAULT_ENV']
if envs == 'base':
    print(envs)
    torch.utils.cpp_extension.CUDA_HOME = '/usr/local/cuda-9.2/'
else:
    print(envs)
    torch.utils.cpp_extension.CUDA_HOME = '/usr/local/cuda-11.7/'
# for glue benchmark
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}
metric_to_eval = {
    "cola": "matthews_correlation",
    "mnli": "accuracy",
    "mrpc": "accuracy",
    "qnli": "accuracy",
    "qqp": "accuracy",
    "rte": "accuracy",
    "sst2": "accuracy",
    "stsb": "pearson",
    "wnli": "accuracy",

}

logger = logging.getLogger(__name__)
os.environ["WANDB_DISABLED"] = "true"
def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def main(args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    global data_args
    model_args, data_args, training_args, _, _ = args
    seed_torch(training_args.seed)
    training_args.metric_for_best_model = metric_to_eval[data_args.dataset_name]
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    
    from task.glue_fewshot.get_trainer import get_trainer

    trainer,predict_dataset = get_trainer(args)
    # Training
    if training_args.do_train:
        train(trainer)

    #Evaluation
    if training_args.do_eval:
        eval(trainer)
    
    # keywords you want to keep
    keywords_to_keep = ['result']

    # get all files in the folder
    files = [f for f in os.listdir(training_args.output_dir)]

    # files to delete
    files_to_delete = []
    print(files)
    for file_name in files:
        if not any(keyword in file_name for keyword in keywords_to_keep):
            files_to_delete.append(file_name)
    print(files_to_delete,'files_to_delete')
    # delete the files
    if torch.cuda.current_device() ==1:
        for file_name in files_to_delete:
            
            if os.path.isfile(file_name):
                os.remove(training_args.output_dir+file_name)
            else:
                shutil.rmtree(training_args.output_dir+file_name)

    

def train(trainer,resume_from_checkpoint=None,last_checkpoint=None):
    trainable_params = sum(p.numel() for n, p in trainer.model.named_parameters() if p.requires_grad and 'classifier' not in n  )
    for n, p in trainer.model.named_parameters():
        if p.requires_grad and 'classifier' not in n and torch.cuda.current_device() ==1: 
            print(n,p.numel())
    
    percentage = trainable_params/124055040 
    print(percentage*100,"%")
    
    # for n,p in trainer.model.named_parameters():
    #     if p.requires_grad:
    #         print(n)
    # exit()

    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    # with torch.autograd.set_detect_anomaly(True):
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(trainer.train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(trainer.train_dataset))
    
    # if model_args.zj_adapter == True:
    #     exit()
    # else:
    #     trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    # whether to save current model and metrics
    # trainer.save_metrics("train", metrics)
    # trainer.save_state() # will save the trainer state will save_model only save the tokenizer
    peak_memory = (torch.cuda.max_memory_allocated() / 1024 ** 2)/1000
    print(
        "Memory utilization",
        peak_memory,
        "GB"
    )

def eval(trainer):
    with torch.no_grad():
        for name ,param in trainer.model.named_parameters():
            if torch.cuda.current_device() == 1:
                if param.requires_grad:
                    print(torch.count_nonzero(param),name)
    #     if param.requires_grad:
    #         if torch.cuda.current_device()==1:
    #             with torch.no_grad():
    #                 if 'scale' in name:
    #                     print(param)
    #                 print(name,torch.count_nonzero(param),len(param))
    #     if 'intrinsic_parameters' in name:
    #         # print(name,param)
    #         temp =  param.clone().detach().requires_grad_(False).to('cpu')
    #         temp = abs(np.sort(-abs(temp)))
    #         print(temp)
    #         plt.plot(np.arange(len(temp[0])),temp[0])
    #         plt.savefig('distribution.jpg')

    logger.info("*** Evaluate ***")
    # for n,p in trainer.model.named_parameters():
    #     if 'intrinsic' in n:
    #         print(n,p)
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    if data_args.task_name != "mnli":
        eval_datasets = [trainer.eval_dataset]
    if data_args.task_name == "mnli":
        eval_datasets = [trainer.eval_dataset]
        dataset = glue_data(trainer.tokenizer,args)
        tasks.append("mnli-mm")
        eval_datasets.append(dataset.mmdataset)
        combined = {}
    for eval_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", combined if task is not None and "mnli" in task else metrics)

def predict(trainer):
    logger.info("*** Predict ***")

    # Loop to handle MNLI double evaluation (matched, mis-matched)
    tasks = [data_args.task_name]
    # predict_datasets = [predict_dataset]
    predict_datasets=[trainer.eval_dataset]
    if data_args.task_name == "mnli":
        tasks.append("mnli-mm")
        predict_datasets.append(raw_datasets["validation_mismatched"])
        combined = {}

    for predict_dataset, task in zip(eval_datasets, tasks):
        metrics = trainer.predict(predict_dataset=predict_dataset)

        max_predict_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(predict_dataset)
        )
        metrics["eval_samples"] = min(max_predict_samples, len(predict_dataset))

        if task == "mnli-mm":
            metrics = {k + "_mm": v for k, v in metrics.items()}
        if task is not None and "mnli" in task:
            combined.update(metrics)

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", combined if task is not None and "mnli" in task else metrics)
    # if data_args.task_name == "mnli":
    #     tasks.append("mnli-mm")
    #     predict_datasets.append(raw_datasets["test_mismatched"])

    # for predict_dataset, task in zip(predict_datasets, tasks):
    #     # Removing the `label` columns because it contains -1 and Trainer won't like that.
    #     predict_dataset = predict_dataset.remove_columns("label")
    #     # predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    #     predictions = trainer.predict(predict_dataset, metric_key_prefix="predict")
    #     trainer.log_metrics("predict", predictions.metrics)
    #     trainer.save_metrics("predict", predictions.metrics)

if __name__ == "__main__":
    args = get_args()
    main(args)
