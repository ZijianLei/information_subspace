import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss
from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel, RobertaModel
)
import numpy as np
from torch.autograd import grad
class MiRobertaForSequenceClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        config.output_hidden_states = True
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dropout_method = 'none'
        self.ib_dim = config.ib_dim
        self.ib = config.ib
        self.activation = 'tanh'
        self.activations = {'tanh': nn.Tanh(), 'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid()}
        if self.ib and self.ib_dim > 0:
            self.kl_annealing = 'linear'
            self.hidden_dim = config.hidden_dim
            intermediate_dim = (self.hidden_dim+config.hidden_size)//2
            self.mlp = nn.Sequential(
                nn.Linear(config.hidden_size, intermediate_dim),
                self.activations[self.activation],
                nn.Linear(intermediate_dim, self.hidden_dim),
                self.activations[self.activation])
            self.beta = config.beta
            # self.beta = 0
            self.sample_size = config.sample_size
            self.emb2mu = nn.Linear(self.hidden_dim, self.ib_dim)
            self.emb2std = nn.Linear(self.hidden_dim, self.ib_dim)
            self.mu_p = nn.Parameter(torch.randn(self.ib_dim))
            self.std_p = nn.Parameter(torch.randn(self.ib_dim))
            # self.classifier = nn.Linear(self.ib_dim, self.config.num_labels)
            self.classifier = RobertaClassificationHead_ibdim(config)
        else:
            self.classifier = RobertaClassificationHead(config)
            # self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)
            # q: what is the function in this file
            # a:
        self.classifier_w0 = dict()
        # print(self.classifier.named_parameters())
        # exit()
        for n, p in self.classifier.named_parameters():

            self.classifier_w0['classifier.'+n] = p.detach().clone()
        # print(self.classifier_w0)
        # exit()
    def estimate(self, emb, emb2mu, emb2std):
        """Estimates mu and std from the given input embeddings."""
        mean = emb2mu(emb)
        std = torch.nn.functional.softplus(emb2std(emb))
        return mean, std
    
    #计算两个分布的KL散度，这里的两个分布是正态分布，q是估计的分布，p是真实的分布
    def kl_div(self, mu_q, std_q, mu_p, std_p):
        """Computes the KL divergence between the two given variational distribution.\
           This computes KL(q||p), which is not symmetric. It quantifies how far is\
           The estimated distribution q from the true distribution of p."""
        k = mu_q.size(1)
        mu_diff = mu_p - mu_q
        mu_diff_sq = torch.mul(mu_diff, mu_diff)
        logdet_std_q = torch.sum(2 * torch.log(torch.clamp(std_q, min=1e-8)), dim=1)
        logdet_std_p = torch.sum(2 * torch.log(torch.clamp(std_p, min=1e-8)), dim=1)
        fs = torch.sum(torch.div(std_q ** 2, std_p ** 2), dim=1) + torch.sum(torch.div(mu_diff_sq, std_p ** 2), dim=1)
        kl_divergence = (fs - k + logdet_std_p - logdet_std_q)*0.5
        return kl_divergence.mean()

    def reparameterize(self, mu, std):
        batch_size = mu.shape[0]
        z = torch.randn(self.sample_size, batch_size, mu.shape[1]).cuda()
        return mu + std * z

    def get_logits(self, z, mu, sampling_type):
        if sampling_type == "iid": # default
            # print(z.shape)
            logits = self.classifier(z)
            # print(logits)
            # exit()
            mean_logits = logits.mean(dim=0)
            logits = logits.permute(1, 2, 0)
        else:
            mean_logits = self.classifier(mu)
            logits = mean_logits
        return logits, mean_logits


    def sampled_loss(self, logits, mean_logits, labels, sampling_type):
        if sampling_type == "iid":
            # During the training, computes the loss with the sampled embeddings.
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.sample_size), labels[:, None].float().expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
            else:
                loss_fct = CrossEntropyLoss(reduce=False)
                loss = loss_fct(logits, labels[:, None].expand(-1, self.sample_size))
                loss = torch.mean(loss, dim=-1)
                loss = torch.mean(loss, dim=0)
        else:
            # During test time, uses the average value for prediction.
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(mean_logits.view(-1), labels.float().view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(mean_logits, labels)
        return loss
    def store_grad(self):
        
        self.param_keys = [n for n,p in self.named_parameters() if p.requires_grad]
        self.gw_dict = dict().fromkeys(self.param_keys)
        for n, p in self.named_parameters():
            if n in self.param_keys:
                if self.gw_dict[n] is None:
                    self.gw_dict[n] = p.grad.detach().clone()  
                else:
                    self.gw_dict[n] += p.grad.detach().clone()
        # print(self.gw_dict.keys())
        # for k in self.gw_dict.keys():
            # if "weight" in k:
           


    def compute_information_bp(self,num_all_batch,no_bp=False):

        
        # param_keys = [p[0] for p in self.named_parameters() if p.requires_grad]
        delta_w_dict = dict()
        for pa in self.named_parameters():
            if "intrinsic_parameter" in pa[0] :
                # w0 = self.w0_dict[pa[0]]
                w0 = 0
                delta_w = pa[1] - w0
                delta_w_dict[pa[0]] = delta_w
            elif 'classifier' in pa[0]:
                w0 = self.classifier_w0[pa[0]]
                delta_w = pa[1] - w0.to(torch.cuda.current_device())
                delta_w_dict[pa[0]] = delta_w
                # print(delta_w)
        info_dict = dict()

        energy_decay = 0
        for k in delta_w_dict.keys():
            delta_w = delta_w_dict[k]
            # print(k,delta_w)
            self.gw_dict[k] *= 1/num_all_batch
            # delta_w.T gw @ gw.T delta_w = (delta_w.T gw)^2
            info_ = (delta_w.flatten() * self.gw_dict[k].flatten()).sum() ** 2
            if no_bp:
                info_dict[k] = info_.item()
            else:
                info_dict[k] = info_
            energy_decay += info_dict[k] 
        # print(energy_decay)
        return energy_decay

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        sampling_type="iid",
        epoch=1,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    Examples::
        from transformers import RobertaTokenizer, RobertaForSequenceClassification
        import torch
        tokenizer = RobertaTokenizer.from_pretrained('bert-base-uncased')
        model = RobertaForSequenceClassification.from_pretrained('bert-base-uncased')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]
        """
        # output hiddens states == True
        #(last_hidden_states,pooled_output,(hidden_states))
        final_outputs = {}
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        # if torch.cuda.current_device() == 1:
        #     print(outputs)
        # exit()
        
        # pooled_output = outputs[1] # [1] is for pooled output,'pool the model by taking the hidden state corresponding to the first token'
        pooled_output = outputs[0] # last hidden state
        # pooled_output = self.dropout(pooled_output)
        loss = {}

       
        if self.ib_dim > 0:
            # this forward function is for VIB
            pooled_output = self.mlp(pooled_output[:,0,:])
            batch_size = pooled_output.shape[0]
            mu, std = self.estimate(pooled_output, self.emb2mu, self.emb2std)
            mu_p = self.mu_p.view(1, -1).expand(batch_size, -1)
            std_p = torch.nn.functional.softplus(self.std_p.view(1, -1).expand(batch_size, -1))
            kl_loss = self.kl_div(mu, std, mu_p, std_p)
            z = self.reparameterize(mu, std)
            final_outputs["z"] = mu

            if self.kl_annealing == "linear":
                beta = min(1.0, epoch*self.beta)

                 
            sampled_logits, logits = self.get_logits(z, mu, sampling_type)
            
            
            if labels is not None:
                ce_loss = self.sampled_loss(sampled_logits, logits, labels.view(-1), sampling_type)
                loss["loss"] = ce_loss + (beta if self.kl_annealing == "linear" else self.beta) * kl_loss
        else:
            # this forward function is for baseline
            logits = self.classifier(pooled_output)
            if labels is not None:
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        ce_loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    ce_loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    ce_loss = loss_fct(logits, labels)
            loss["loss"] = ce_loss
        final_outputs.update({"logits": logits, "loss": loss["loss"], "hidden_attention": outputs[2:]})
        return final_outputs


class RobertaClassificationHead_ibdim(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.ib_dim, config.ib_dim)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.ib_dim, config.num_labels)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x