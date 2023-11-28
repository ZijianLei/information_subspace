from transformers import Trainer
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
import numpy as np

# the first one is our measurement of the importance
class IIBRankSelector(object):
    def __init__(self,model,total_step=None):
        self.logsoftmax = nn.LogSoftmax()
        self.model = model
        self.total_step = total_step
        self.selection_step = None
        self.grad_value = {}
        self.delta_grad_value = {}
        self.information_score = {}
        self.delta_information_score = {}
        self.mask_dict = {}
        self.delta_mask = []
        self.temp_name = None
        self.target_id = None
        self.mask_threshold = None
        self.gw_dict = dict()
        self.target_id_rate = None
        self.w_init = {}
        self.temp_mask = None
        self.information_plot = torch.zeros(12,1)
        self.information_prob_plot = torch.zeros(12,1)
        self.covariance_plot = {}
        self.sizes = {}
        self.all_params_size = 0
        # self.plot_prob = torch.zeros(12,1)
        self.select_count = None
        self.epsilon = None # hyperparamter control the trade-off between exploration and exploitation
        self.temperature = 1e-5 #1e-3  # default to be 1,use to change the distribution
        # model.module.low_rank_embdding = dict()
        self.low_rank_dim = 100
        self.pca_w = {}
        self.Z = {}
        self.mean_parameter = {}
        self.info_grad_value = {}
        self.egen_value = {}
        self.error_rate = 0.1
    def set_total_step(self,total_step):
        self.total_step = total_step

    def update_ipt(self,model):
        for n,p in model.named_parameters():
            
            if 'intrinsic_param' in n:     
                if n not in self.grad_value:
                    self.grad_value[n] = torch.zeros_like(p)
                    self.length = len(p)
                if n not in self.w_init:
                    self.w_init[n] = None
                    # self.mask_self.exploration_function(current_step=global_step)[n] = torch.zeros_like(p)
                with torch.no_grad():
                    self.grad_value[n] = p.grad.detach()
                    # self.delta_grad_value[n] += p.grad.detach()
                    # self.grad_value[n] += (p.grad.detach())
        self.size = torch.zeros(len(self.grad_value)*self.length)
        
        if self.select_count == None:
            self.select_count = torch.ones_like(self.size)
            self.last_select_count = torch.zeros_like(self.size)
            self.epsilon = 1/(len(self.grad_value)*self.length)
            self.epsilon = -1
            self.plot_delta_mask = torch.zeros((len(self.grad_value)*self.length,1))
            self.plot_mask = torch.zeros_like(self.plot_delta_mask)
    def enable_grad(self,model):
        for k,v in model.named_parameters():
            if 'sparse_mask' in k and 'embeddings' not in k:
                # print(k)
                v.requires_grad_()
                

    def store_grad(self,model):
        for n,p in model.named_parameters():
            
            if 'sparse_mask' in n and 'bias' not in n and p.grad != None:
                
                if n not in self.info_grad_value:
                    # print(p.requires_grad)
                    self.info_grad_value[n] = p.grad.clone().detach()
                else:
                    self.info_grad_value[n] +=p.grad.clone().detach()


    def create_intrinsic_mask(self,model,global_step):
        #  first is the global warm-up step
        # global_step -= self.selection_step
        self.target_id = model.module.mask_target_number
        self.selection_step = model.module.selection_step
        
        
            # print(self.info_grad_value,'info_grad')
        
        for n,p in model.named_parameters():
            
            if  n in self.info_grad_value:
                    
                
                self.delta_information_score[n] = self.info_grad_value[n]**2
                self.sizes[n] = p.shape
                self.delta_mask.append(self.delta_information_score[n].view(-1))
                self.all_params_size += torch.prod(torch.tensor(p.shape))
                self.w_init[n] = p.data.detach().clone()
                # self.delta_grad_value[n] = torch.zeros_like(p)
                    # self.information_score[n] = self.grad_value[n]*p.data.detach()*(self.grad_value[n]*p.data.detach()).sum()
                    # self.mask.append(self.delta_information_score[n])
            
                # using the moving average
                self.beta = 0.85

        self.delta_mask = torch.cat(self.delta_mask,0)
        
        
        
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']
        # self.probability = nn.functional.softmax(torch.abs(self.mask)/self.temperature,dim=0).cpu()
        _,ind = torch.topk(torch.abs(self.delta_mask),self.target_id)
        
        self.temp_mask = torch.zeros_like(self.delta_mask)       
        self.temp_mask[ind] = 1
        assert self.temp_mask.long().sum() == self.target_id
        now_idx = 0
        for k,v in self.sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            self.mask_dict[k] = self.temp_mask[now_idx:end_idx].reshape(v).to(torch.cuda.current_device())
            now_idx = end_idx
        return self.mask_dict
        
            
    def grad_drop(self,model,global_step):  
        if self.temp_mask !=None:
            
            # print(self.temp_mask)      
            for n,p in model.named_parameters():
                if n in self.mask_dict:
                       
                    p.grad.data.copy_(p.grad.data * self.mask_dict[n])
                    
                    
                

    def compute_low_rank_embedding(self,model):
        for n,p in model.named_parameters():
            if 'intrinsic_param' in n: 
                idx = [int(s) for s in re.findall(r"\d+",n)]
                idx =str(idx[0])
                # print(model.module.low_rank_embedding)
                # exit()
                # if n not in model.module.low_rank_embedding:
                #     model.module.low_rank_embedding[idx] = torch.zeros(len(p),self.selection_step)
                    # self.Z[n] = torch.zeros(len(p),self.selection_step) # the candidate set
                index = torch.where(model.module.low_rank_embedding[idx].sum(dim=0)==0)[0]
                x = p.data.detach().clone().reshape(-1)
                if index.nelement()!=0:
                    # print(x.size(),model.module.low_rank_embedding[idx][:,index[0]].size())
                    model.module.low_rank_embedding[idx][:,index[0]] +=  x   
                else:
                
                    u,_,_ = torch.svd_lowrank(model.module.low_rank_embedding[idx],q=self.low_rank_dim)
                    model.module.low_rank_embedding[idx][:,:self.low_rank_dim] = u
                    model.module.low_rank_embedding[idx][:,self.low_rank_dim:] = 0
                
                
                
                
    def compute_intrinsic_grad(self,model):
        
        for n, p in model.named_parameters():
            if "intrinsic_param" in n:
                idx = [int(s) for s in re.findall(r"\d+",n)]
                idx =str(idx[0])
                # model.module.low_rank_embedding[idx] = model.module.low_rank_embedding[idx].to(torch.cuda.current_device())
                p.grad.data = model.module.low_rank_embedding[idx][:,:self.low_rank_dim]@torch.t(model.module.low_rank_embedding[idx][:,:self.low_rank_dim])@p.grad.data
    

    def create_intrinsic_projection(self,model,global_step):
        if global_step < self.selection_step:
            self.compute_low_rank_embedding(model)
        else:
            if global_step % 2 ==0:
                self.compute_low_rank_embedding(model)
            self.compute_intrinsic_grad(model)
                        
