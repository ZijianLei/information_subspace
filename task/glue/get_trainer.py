
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
from datasets import load_dataset, load_metric
import re
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoAdapterModel,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    TrainingArguments,
    default_data_collator,
    set_seed,
    AdapterTrainer,
    AdapterConfig,
    
)
# from my_model.Trainer import Trainer

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from task.glue.glue_datasets import glue_data
from module.modeling_ibroberta import  *
from adapter_model import *
from module.subspace_selector import IIBRankSelector
def get_trainer(args,basis = None):
    model_args, data_args, training_args, _, adapter_args = args
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )

    dataset = glue_data(tokenizer,args)
    if not dataset.is_regression:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            label2id=dataset.label2id,
            id2label=dataset.id2label,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            finetuning_task=data_args.dataset_name,
            revision=model_args.model_revision,
        )
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
    
    
    
    # the adapter part to be finish
    # Setup adapters
    if adapter_args.train_adapter:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # init an adapter model
        model = AutoAdapterModel.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model.add_classification_head(
            data_args.dataset_name,
            num_labels=dataset.num_labels,
            # id2label={i: v for i, v in enumerate(label_list)} if num_labels > 0 else None,
        )
        model = adapter_model(model, args)
    elif model_args.zj_adapter or model_args.said:
        # this is tmp code for intrinsic tuning, will merged to adapter in the future
        
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=dataset.num_labels,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # PET with PAC-Bayes information bottleneck regularizatiion
        if model_args.ib ==True and model_args.ib_dim == 0:
            config.ib = model_args.ib
            config.ib_dim = model_args.ib_dim
            training_args.ib_beta = model_args.beta
            training_args.noise_scale =  model_args.noise_scale
            model = MiRobertaForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                )
           
        else:
            # PET
            model = AutoModelForSequenceClassification.from_pretrained(
            # model = AutoAdapterModel.from_pretrained(   
                model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                use_auth_token=True if model_args.use_auth_token else None,
                # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                )
        # get the adapter model
        model = adapter_model(model, args,basis,config)
        
    elif model_args.ib and model_args.ib_dim > 0:
        # finetune the pretrained model via vib
        config.ib = model_args.ib
        config.ib_dim = model_args.ib_dim
        config.hidden_dim = (768+model_args.ib_dim) //2
        config.sample_size = 5
        model = MiRobertaForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
            )
        
    else:
        #JUST finetuning without information bottleneck regularization
        # or finetune with model ib, will use the origin model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )
       
  
 
     # Initialize our Trainer
    if  model_args.ib or (model_args.zj_adapter and 'ib_adapter' in training_args.output_dir):
        #using the customized trainer
        from module.id_trainer import Trainer
        optimizer_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if 'classifier' not in n and p.requires_grad],
                "weight_decay": training_args.weight_decay,
                "lr": training_args.learning_rate
            },
            {
                "params": [p for n, p in model.named_parameters() if 'classifier' in n and p.requires_grad],
                "weight_decay": training_args.weight_decay,
                "lr": training_args.learning_rate,
            }
        ]
        
        optimizer = torch.optim.AdamW(optimizer_parameters)
        subselector = IIBRankSelector(model)
        trainer_class = Trainer
        trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset if training_args.do_train else None,
        eval_dataset=dataset.eval_dataset if training_args.do_eval else None,      
        compute_metrics=dataset.compute_metrics,
        tokenizer=tokenizer,
        data_collator=dataset.data_collator,
        subselector = subselector,
        # optimizers=(optimizer,None)
        )
    else:
        from transformers import Trainer
        trainer_class = AdapterTrainer if adapter_args.train_adapter else Trainer
        # trainer_class = AdapterTrainer
        trainer = trainer_class(
            model=model,
            args=training_args,
            train_dataset=dataset.train_dataset if training_args.do_train else None,
            eval_dataset=dataset.eval_dataset if training_args.do_eval else None,      
            compute_metrics=dataset.compute_metrics,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            # optimizer=
        )

    return trainer, dataset.predict_dataset

