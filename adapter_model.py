from transformers.adapters import (
    ConfigUnion,
    AdapterConfig,
)
from transformers import (
    ParallelConfig,
    CompacterConfig,
    LoRAConfig,
    PfeifferConfig,
)
# from my_model import *

import re

# from my_model.masking_subspace import *
from variation_hadamard_subspace import *

import torch
def adapter_model(model,args,basis=None,config=None):
    model_args, data_args, training_args, _, adapter_args = args
    # if 'local' in training_args.output_dir:
    #     from my_model.adaptive_subspace_id import *
    # else:
    #     from my_model.id_adapter import *
    # using which adapter -> model_args.xxx , the baseline adapters
    if adapter_args.train_adapter == True: 
        if model_args.compactor == True:
            config = CompacterConfig(phm_dim=model_args.adapter_phm_dim)
            # model.add_adapter(data_args.dataset_name,config = config,overwrite_ok = True)
        elif model_args.parallel_adapter == True:
            config = ParallelConfig(reduction_factor=adapter_args.adapter_reduction_factor)
        elif model_args.sequencial_adapter==True:
            config = PfeifferConfig()
        elif model_args.lora ==True:
            config  = LoRAConfig()
        # elif model_args.lora ==True:
        #     config = 
        model.add_adapter(data_args.dataset_name,config = config,  overwrite_ok = True )
        # Freeze all model weights except of those of this adapter
        model.train_adapter([data_args.dataset_name])  
        # Set the adapters to be used in every forward pass
        model.set_active_adapters([data_args.dataset_name])



    # my adapter_achitechture
    elif model_args.zj_adapter == True:
        
        filter_set = set()
        if model_args.direction_selection_rate !=0:
            # model = masked_id_adapter(model, model_args.intrinsic_dim, training_args.output_dir, config = config,str_filter=filter_set, projection = "orthogonal", device = 0,general_subspace_rate = model_args.direction_selection_rate,ib_args = model_args)
            model = masked_id_adapter(model, model_args.intrinsic_dim,
            training_args.output_dir, config = config,str_filter=filter_set, projection = "orthogonal", device = 0,general_subspace_rate = model_args.direction_selection_rate,ib_args = model_args)
            print('lid direction selection')          
        else: 
            # model = masked_id_adapter(model, model_args.intrinsic_dim,
            # training_args.output_dir, config = config,str_filter=filter_set, projection = "orthogonal", device = 0,general_subspace_rate = model_args.direction_selection_rate)
            model = masked_id_adapter(model, model_args.intrinsic_dim,
            training_args.output_dir, config = config,str_filter=filter_set, projection = "random", device = 0,general_subspace_rate = model_args.direction_selection_rate)
            # model = id_adapter(model, model_args.intrinsic_dim,
            # training_args.output_dir, config = config,str_filter=filter_set, projection = "orthogonal", device = 0)
            print('original lid')


    

    return model

