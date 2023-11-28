'''
This verision is sketch and project
'''
import torch
import torch.nn as nn
from sys import getsizeof as getsize
import torch.nn.functional as F
from transformers import (
    AdapterLayer,
    AdapterConfig,

)
import re
import torch.cuda.comm
import numpy as np
# from fwh_cuda import fast_walsh_hadamard_transform
from typing import Tuple, Set
import sys
from transformers.utils.model_parallel_utils import assert_device_map,get_device_map
sys.setrecursionlimit(60265)
import torch.utils.cpp_extension
import os# detect the DAAI running enviroment
# import matplotlib.pyplot as plt
# import seaborn as sns
envs = os.environ['CONDA_DEFAULT_ENV']
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
if envs == 'base':
    torch.utils.cpp_extension.CUDA_HOME = '/usr/local/cuda-12.2/'
else:
    print(envs)
    torch.utils.cpp_extension.CUDA_HOME = '/usr/local/cuda-11.7/'
hadamard_cuda = torch.utils.cpp_extension.load(
    name='cuda_hadamard',
    sources=[
        'fwh/cuda_hadamard.cpp',
        'fwh/cuda_hadamard_kernel.cu',
    ],
    extra_cuda_cflags=['-O2'],
    verbose=False
    )

def fastfood_vars(DD, device=0):
    """
    Returns parameters for fast food transform
    :param DD: desired dimension
    :return:
    """
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll

    # Binary scaling matrix where $B_{i,i} \in \{\pm 1 \}$ drawn iid
    BB = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB = (BB * 2 - 1)
    BB.requires_grad_(False)

    # Random permutation matrix
    Pi = torch.LongTensor(np.random.permutation(LL))
    Pi.requires_grad_(False)

    # Gaussian scaling matrix, whose elements $G_{i,i} \sim \mathcal{N}(0, 1)$
    GG = torch.FloatTensor(LL,).normal_()
    GG.requires_grad_(False)
    divisor = torch.sqrt(LL * torch.sum(torch.pow(GG, 2)))
    return [BB.to(device), Pi.to(device), GG.to(device), divisor.to(device), LL]



def orthogonal_vars(DD, device = 0):
    """
    Returns parameters for structure orthogonal transform
    :param DD: desired dimension
    :return:
    """
    # othogoanl random fast matrix
    ll = int(np.ceil(np.log(DD) / np.log(2)))
    LL = 2 ** ll
    # torch.manual_seed(111)
    BB1 = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB1 = (BB1 * 2 - 1)
    BB1.requires_grad_(False)
    
    # torch.manual_seed(222)
    BB2 = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    BB2 = (BB2 * 2 - 1)
    BB2.requires_grad_(False)
    # BB3 = torch.FloatTensor(LL).uniform_(0, 2).type(torch.LongTensor)
    # BB3 = (BB2 * 2 - 1)
    # BB3.requires_grad_(False)
    divisor = torch.sqrt(LL*torch.sum(torch.ones(1,)))
    divisor.requires_grad_(False)
    return [BB1.to(device), BB2.to(device),divisor.to(device), LL]
    # divisor = torch.sqrt(LL*torch.sum(torch.ones(1,)))
    # divisor.requires_grad_(False)
    # block_seed = torch.randint(1000,(1,))[0]
    # return [block_seed.to(device),divisor.to(device), LL]

class HadamardTransformCuda(torch.autograd.Function):
    '''The unnormalized Hadamard transform (i.e. without dividing by sqrt(2))
    '''
    @staticmethod
    def forward(ctx, u):
        return hadamard_cuda.hadamard_transform(u)

    @staticmethod
    def backward(ctx, grad):

        return HadamardTransformCuda.apply(grad)


class Masked_ID_adapter:
    def __init__(self, module: nn.Module,intrinsic_dimension:int, output_dir,config,
    str_filter:Set[str] = set(), projection='orthogonal',device='cpu',general_subspace_rate = 1,ib_args = None):
        self.classifier_dict = dict()
        self.general_subspace_rate = general_subspace_rate
        self.module = module
        self.device = torch.cuda.current_device()
        self.merged = False  # default to be false, set to be ture when merge the parameter in inference stage, will remove the same parameter
        # for base:12 large:24
        self.call_count = 0
        self.reinitialized = 0
        self.mask = None # initial the gradient mask
        # self.transformer_num = config.num_hidden_layers
        self.name_base_localname_idx = []
        self.intrinsic_dimension = intrinsic_dimension
        self.idx_list =[]
        self.initial_value = dict()
        self.projection_params = {}
        self.projection = projection
        self.low_rank_dim = 200
        # self.plot_prob = torch.zeros(config.num_hidden_layers,1)
        # self.intrinsic_parameters = nn.ParameterDict()
        # 
        self.dropout_fn = nn.Dropout(p=0.1)
        self.intrinsic_parameters = nn.ParameterDict()
        # self.sparse_mask = nn.ParameterDict()

        # self.mask = dict()
        # self.fisher = dict()
        self.dropout_mask = None
        self.low_rank_embed_dict = nn.ParameterDict()
        self.dropout_method = ib_args.dropout_method       
        self.selection_step = ib_args.selection_step
        self.mask_target_number = None
        # module.register_parameter("intrinsic_parameters", self.intrinsic_parameters)
        setattr(module, "intrinsic_parameters", self.intrinsic_parameters)
        setattr(module, "dropout_mask", self.dropout_mask)
        setattr(module,"low_rank_embedding",self.low_rank_embed_dict)
        setattr(module,'general_subspace_rate',self.general_subspace_rate)
        
        setattr(module,'dropout_method',self.dropout_method)
        setattr(module,'selection_step',self.selection_step)
        # setattr(module,'sparse_mask',self.sparse_mask)
        
        block_count = config.num_hidden_layers
        # block_size = int(self.intrinsic_dimension/block_count)
        # self.block_size = block_size
        
        if self.general_subspace_rate !=0:
            # selection number is currently defined for 
            self.candidate_number = int(self.general_subspace_rate*self.intrinsic_dimension/block_count) 
            self.mask_target_number = self.intrinsic_dimension - self.candidate_number*block_count
            self.mask_target_number = 1000
        else:
            
            self.mask_target_number = self.intrinsic_dimension
        setattr(module,'mask_target_number',self.mask_target_number)
        for name , param in list(module.named_parameters()):   
            # if torch.cuda.current_device() == 1:
            #     print(name)
            if  (len(str_filter) == 0 or any([x in name for x in str_filter  ])) and 'classifier' not in name and 'mlp' not in name and 'emb2' not in name:#and 'embeddings' not in name:
                
                self.initial_value[name] = v0 = (
                    param.clone().detach().requires_grad_(False).to(self.device)
                )
                # self.sparse_mask[name.replace(".","_")] = nn.Parameter(torch.zeros_like(param),requires_grad=False)
                init_shape = self.initial_value[name].size()
                DD = np.prod(v0.size())
                basis_size = DD
                # if basis_size not in self.projection_params:
                #     self.projection_params[basis_size] = self.get_projection_params(DD, self.device)
                # self.projection_params[name]  =self.get_projection_params(DD,self.device)
                idx = [int(s) for s in re.findall(r"\d+",name)]
                if len(idx)!=0:
                    
                    idx =str(idx[0])  
                    name_prefix,block_name = name.split(idx,1)
                    
                    if block_name not in self.projection_params:
                        self.projection_params[block_name] = self.get_projection_params(DD,self.device)
                        # self.projection_params[block_name] = torch.randint(1000,(1,))[0]
                    if idx not in self.intrinsic_parameters.keys():                        
                        self.intrinsic_parameters.update({idx:nn.Parameter(torch.zeros(self.candidate_number),requires_grad=True)}) 
                        # self.low_rank_embed_dict.update({idx:nn.Parameter(torch.zeros(self.candidate_number,self.low_rank_dim),requires_grad=False)})
                    if (idx,basis_size,init_shape,block_name) not in self.idx_list:
                        self.idx_list.append((idx,basis_size,init_shape,block_name))

                base, localname = module, name
                # until here, no "sparse_mask" in name
                while "." in localname:  
                    
                    prefix, localname = localname.split(".", 1)                 
                    base = base.__getattr__(prefix)               
                self.name_base_localname_idx.append((name, base, localname,basis_size))
                
            if "classifier" not in name and 'mlp' not in name and 'emb2' not in name: # use backpropagation to update theclassifier   
                    # print(name)                    
                param.requires_grad_(False)
        
        # w0_dict = dict()   
        # for name , param in list(module.named_parameters()):
        #     w0_dict[name] = param.clone().detach()
        # module.w0_dict = w0_dict

        length = len(self.initial_value)
        
        self.intrinsic_scaler = nn.Parameter(torch.ones((length)).cuda())
        module.register_parameter(
                "intrinsic_scaler", self.intrinsic_scaler)
        setattr(module, "intrinsic_scaler",
                self.intrinsic_scaler)



    def get_projection_params(self,DD,device):
        if self.projection == 'orthogonal':
            return orthogonal_vars(DD,device)
        else:
            return fastfood_vars(DD,device)

    def get_projected_param(self, intrinsic_vec, DD, projection_params):
        if  self.projection == "orthogonal":
            ray =  self.orthogonal_torched(intrinsic_vec, DD, projection_params)
            # print(fisher)
        else:
            ray = self.fastfood_torched(intrinsic_vec, DD, projection_params)
        return ray

    @staticmethod
    def apply(module:nn.Module,intrinsic_dimension:int, output_dir,
                str_filter:Set[str] = set(), projection='fastfood',device='cuda',transformer_num: int =12,general_subspace_rate = 1,ib_args = None):
        # print(projection)
        # for checking no pre hooks
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, Masked_ID_adapter) and hook.name == name:
                raise RuntimeError("Cannot register two intrinsic dimension hooks on "
                                "the same parameter {}".format(name))
        #initial the Masked_ID_adapter
        fn = Masked_ID_adapter(
            module, intrinsic_dimension, output_dir, str_filter,  projection, device,transformer_num,general_subspace_rate,ib_args = ib_args)

        
        module.register_forward_pre_hook(fn) # adding global state to the nn module   
        
        return fn
    


    def __call__(self,module,  inputs):
        index = 0

          
        with torch.enable_grad():

                                    
            for name, base, localname,basis_size in self.name_base_localname_idx:
                # delete some encoding part
                if 'embeddings' in name or 'LayerNorm' in name:
                    
                    param = self.initial_value[name]
                    
                    delattr(base, localname) 
                    setattr(base, localname, param)
                    continue
                temp_idx = [int(s) for s in re.findall(r"\d+",name)]
                if len(temp_idx)!=0:    
                    
                                 
                    temp_idx =str(temp_idx[0])  
                    # print(module.training)
                    x = module.intrinsic_parameters[temp_idx]
                    if module.training: 
                        # only normal dropout
                        x = self.dropout_fn(module.intrinsic_parameters[temp_idx])
                    
                        
                    name_prefix,block_name = name.split(temp_idx,1)
                    embedding_size = np.prod(self.initial_value[name].size())
                    ray= self.get_projected_param(x, embedding_size, self.projection_params[block_name])
                # if idx !=temp_idx or embedding_size!=basis_size:
                #     continue
                
                
                # if block_name in name:
                    
                    index = list(self.initial_value).index(name)
                    # print(self.sparse_mask[name.replace(".","_")])
                    # if torch.cuda.current_device() ==1:
                        # print(torch.count_nonzero(self.sparse_mask[name.replace(".","_")].clone().detach())) 
                    param = self.initial_value[name]+self.intrinsic_scaler[index]*ray.reshape_as(self.initial_value[name])
                    # param = self.initial_value[name]+ self.sparse_mask[name.replace(".","_")]+self.intrinsic_scaler[index]*ray.reshape_as(self.initial_value[name])
                    delattr(base, localname)    
                    setattr(base, localname, param)

    def fastfood_torched(self,x, DD: int, param_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]):
        """
        Fastfood transform
        :param x: array of dd dimension
        :param DD: desired dimension
        :return:
        """
        dd = x.size(0)

        BB, Pi, GG, divisor, LL = param_list
        # Padd x if needed
        dd_pad = F.pad(x, pad=(0, LL - dd), value=0.0, mode="constant")
        # From left to right HGPiH(BX), where H is Walsh-Hadamard matrix
        dd_pad = dd_pad * BB

        # HGPi(HBX)
        mul_2 =  HadamardTransformCuda.apply(dd_pad)

        # HG(PiHBX)
        mul_3 = mul_2[Pi]

        # H(GPiHBX)
        mul_3 = mul_3 * GG

        # (HGPiHBX)
        mul_5 = HadamardTransformCuda.apply(mul_3)

        ret = mul_5[:int(DD)]
        ret = ret / \
            (divisor * np.sqrt(float(DD) / LL))
        return ret
           
    '''
    The following part is for fisher information and importance estimation

    '''
    def orthogonal_torched(self,x, DD: int, param_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]):

        dd = x.size(0)
        # x = x.to(torch.cuda.current_device())

        '''
        remaining problem memory explore when fisher info accm
        '''

        BB1, BB2, divisor, LL = param_list
        
        # padding_size = DD - x.size(0)
        # dd_concat = F.pad(x,pad=(0,padding_size),value=0.0,mode='constant')
            # print(torch.count_nonzero(dd_concat.detach()),torch.count_nonzero(x.detach()))
        padding_size = LL-x.size(0)
        dd_concat = F.pad(x,pad=(0,padding_size),value=0.0,mode='constant') 

        
        # dd2 = torch.count_nonzero(dd_concat.detach())
        
        dd_concat = dd_concat * BB1
 
        dd_concat= HadamardTransformCuda.apply(dd_concat)           
        dd_concat =(1/np.sqrt(LL))*dd_concat* BB2
        dd_concat = HadamardTransformCuda.apply(dd_concat)
        ret = (1/np.sqrt(LL))*dd_concat[:int(DD)]
        ret = ret/(np.sqrt(float(DD)/LL))
        return ret
    # def orthogonal_torched(self,x, DD: int, param_list: Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor, int]):

    #     dd = x.size(0)
    #     # x = x.to(torch.cuda.current_device())

    #     '''
    #     remaining problem memory explore when fisher info accm
    #     '''
    #     device = torch.cuda.current_device()
    #     block_seed, divisor, LL = param_list
        
    #     # padding_size = DD - x.size(0)
    #     # dd_concat = F.pad(x,pad=(0,padding_size),value=0.0,mode='constant')
    #         # print(torch.count_nonzero(dd_concat.detach()),torch.count_nonzero(x.detach()))
    #     torch.manual_seed(block_seed)
    #     padding_size = LL-x.size(0)
    #     dd_concat = F.pad(x,pad=(0,padding_size),value=0.0,mode='constant') 

        
    #     # dd2 = torch.count_nonzero(dd_concat.detach())
        
    #     dd_concat = dd_concat * torch.randint(2,dd_concat.size()).to(device)
 
    #     dd_concat= HadamardTransformCuda.apply(dd_concat)           
    #     dd_concat =(1/np.sqrt(LL))*dd_concat* torch.randint(2,dd_concat.size()).to(device)
    #     dd_concat = HadamardTransformCuda.apply(dd_concat)
    #     ret = (1/np.sqrt(LL))*dd_concat[:int(DD)]
    #     ret = ret/(np.sqrt(float(DD)/LL))
    #     return ret




        
def masked_id_adapter(module, intrinsic_dimension,  output_dir,config, str_filter, projection, device="cuda",general_subspace_rate = 1,ib_args = None):
    print(projection)
    Masked_ID_adapter.apply(
        module, intrinsic_dimension,  output_dir, config,str_filter,  projection, device,general_subspace_rate,ib_args)
    # module = nn.DataParallel(module)
    return module




