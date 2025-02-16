from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
import torch_geometric.transforms as T
from torch_geometric.loader import NeighborLoader
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import torch
import copy
import argparse
import numpy as np
import json
import scipy
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup,LlamaTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, LoraModel, PeftConfig, PeftModel
import os
import pickle
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import json
import random
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
import pickle
from proj import FP
import random


def get_total_grad_norm(parameters):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def get_first_and_second_order_neighbors(data, input_ids):
    # 将邻接矩阵转换为 SparseTensor，这里直接使用 data.adj_t，因为它已经是 SparseTensor 类型
    adj_matrix = data.adj_t
    
    # 创建一个从原始节点 ID 到新节点 ID 的映射
    id_mapping = {old_id.item(): new_id for new_id, old_id in enumerate(data.n_id)}
    
    # 创建一个空字典来存储结果
    neighbors_dict = {}
    
    # 对于每一个输入节点
    for node_id in input_ids:
        # 将原始节点 ID 映射到新的节点 ID
        new_node_id = id_mapping[node_id.item()]
        
        # 获取该节点的所有邻居
        first_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == new_node_id]
        
        # 获取一阶邻居的原始 ID
        first_order_neighbors = {data.n_id[i].item(): [] for i in first_order_neighbor_ids.tolist()}
        
        # 为每个一阶邻居获取二阶邻居
        for first_order_neighbor in first_order_neighbor_ids:
            # 获取该一阶邻居的所有邻居
            second_order_neighbor_ids = adj_matrix.storage.col()[adj_matrix.storage.row() == first_order_neighbor]
            
            # 过滤掉自己作为一阶邻居的情况
            second_order_neighbor_ids = second_order_neighbor_ids[second_order_neighbor_ids != new_node_id]
            
            # 获取二阶邻居的原始 ID
            second_order_neighbors = [data.n_id[i].item() for i in second_order_neighbor_ids.tolist()]
            
            # 添加二阶邻居到对应的一阶邻居下
            first_order_neighbors[data.n_id[first_order_neighbor].item()].extend(second_order_neighbors)
        
        # 将邻居列表添加到字典中
        neighbors_dict[node_id.item()] = first_order_neighbors
    return neighbors_dict

def get_args():
    parser = argparse.ArgumentParser(description="PyTorch PYG implementation")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    # CPU/GPU
    parser.add_argument("--num_workers", default=6, type=int)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    
    # LLM Config
    parser.add_argument('--backbone', type=str, default='./llama2-7b-hf')
    parser.add_argument('--tokenizer', type=str, default='LlamaTokenizer')
    parser.add_argument('--max_text_length', type=int, default=4096)
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=128)
    parser.add_argument('--lora_dropout', type=int, default=0.05)
    
    # LLM Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.05)
    parser.add_argument('--weight_decay', type=float, default=0.005)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-8)
    parser.add_argument('--adam_beta1', type=float, default=0.9)
    parser.add_argument('--adam_beta2', type=float, default=0.999)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        "--num_neighbors",
        type=str,
        default="15,15",
        help="Number of samples for each layer in SAGE. Length = num_layers",
    )
    parser.add_argument(
        "--num_edges",
        type=str,
        default="15",
        help="Number of edges for each layer. Length = num_layers",
    )

    """Dataset"""
    parser.add_argument('--do_lower_case', action='store_true')
    parser.add_argument("--dataset", type=str, default="amazon_ratings", help="Dataset")
    parser.add_argument("--data_path", type=str, default="./data", help="Path to data")
    parser.add_argument("--num_nodes", type=int, default="24492", help="the number of nodes")
    
    """Global """
    parser.add_argument("--train", type=bool, default="True", help="training ")
    parser.add_argument("--test", type=bool, default="False", help="testing ")
    args = parser.parse_args(args=[])

    return args

def pre_data(args):
    if args.dataset=='ogbn-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
        data=dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        split_idx = dataset.get_idx_split()
        train_loader = NeighborLoader(data, input_nodes=split_idx["train"],
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                      ,batch_size=args.batch_size, 
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=split_idx["valid"],
                                      batch_size=args.batch_size,
                                         num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                      , shuffle=False,num_workers=args.num_workers)
        test_loader = NeighborLoader(copy.copy(data), input_nodes=split_idx["test"],
                                     batch_size=args.batch_size,
                                    num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")]
                                     , shuffle=False,num_workers=args.num_workers)
    if args.dataset=='deezer-europe':
        deezer = scipy.io.loadmat(f'./deezer_europe/deezer-europe.mat')
        adj_t= SparseTensor(row=torch.tensor(deezer['A'].tocoo().row).to(torch.long), col=torch.tensor(deezer['A'].tocoo().col).to(torch.long),sparse_sizes=(len(deezer['label'][0]), len(deezer['label'][0])))
        data= Data(x=torch.tensor(deezer['features'].toarray()).to(torch.float32), adj_t=adj_t,y=torch.tensor(deezer['label']).squeeze())
        data.adj_t = data.adj_t.to_symmetric()
        # 获取节点总数
        num_nodes = len(data.y)
        # 定义数据集划分比例
        train_ratio = 0.5
        val_ratio = 0.25
        test_ratio = 0.25
        # 计算每种数据集包含的节点数
        num_train = int(num_nodes * train_ratio)
        num_val = int(num_nodes * val_ratio)
        num_test = num_nodes - num_train - num_val
        # 随机排列节点索引
        node_indices = torch.randperm(num_nodes)
        # 切分索引
        train_indices = node_indices[:num_train]
        val_indices = node_indices[num_train:num_train + num_val]
        test_indices = node_indices[num_train + num_val:]

        train_loader = NeighborLoader(data, input_nodes=train_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)    
    if args.dataset in ['roman_empire','amazon_ratings','questions']:
        file_path = f'./{args.dataset}/{args.dataset}_right.npz'
        data = np.load(file_path)
        
        # 切分索引
        train_indices = np.where(data['train_masks'][0])[0]
        val_indices = np.where(data['val_masks'][0])[0]
        test_indices = np.where(data['test_masks'][0])[0]
        
        
        # data = np.load('./roman_empire/roman_empire.npz')
        adj_t= SparseTensor(row=torch.tensor(data['edges']).t()[0].to(torch.long), col=torch.tensor(data['edges']).t()[1].to(torch.long),sparse_sizes=(len(data['node_labels']),len(data['node_labels']) ))
        data= Data(x=torch.tensor(data['node_features']), adj_t=adj_t,y=torch.tensor(data['node_labels']))
        data.adj_t = data.adj_t.to_symmetric()


        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
        
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
        num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                batch_size=args.batch_size, shuffle=False,num_workers=args.num_workers)
    if args.dataset in ['wikics']:
        file_path = f'./{args.dataset}/data.pt'
        data = torch.load(file_path)
        
        node_id = np.arange(data.num_nodes)
            
        train_indices=node_id[data['train_mask'][:,0]]
        val_indices=node_id[data['val_mask'][:,0]]
        test_indices=node_id[data['test_mask']]
         
        train_loader = NeighborLoader(data, input_nodes=train_indices,
                                       num_neighbors=[eval(num_neighbors) for num_neighbors in args.num_neighbors.split(",")], 
                                      batch_size=args.batch_size,
                                      shuffle=True,num_workers=args.num_workers,
                                      pin_memory=True)
        valid_loader = NeighborLoader(copy.copy(data), input_nodes=val_indices,
                                                  num_neighbors=[-1], shuffle=False)
        test_loader = NeighborLoader(copy.copy(data), input_nodes=test_indices,
                                                  num_neighbors=[-1], shuffle=False)
    return train_loader,valid_loader,test_loader,data

# template={}
# template['train']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that connect {}, what is the word {} syntactic role? <Assistant>: {}"
# template['test']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that connect {}, what is the word {} syntactic role? <Assistant>:"

# template={}
# template['train']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that {} {}, what is the word {} syntactic role? <Assistant>: {}"
# template['test']="<User>: In an article, words that have dependency relationships (where one word depends on another) are connected, forming a dependency graph. Based on the connections between words, determine the syntactic role of each word. Given that a word {} that {} {}, what is the word {} syntactic role? <Assistant>:"

# template={}
# template['train']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that {} {}, what is the product {} rating? <Assistant>: {}"
# template['test']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that {} {}, what is the product {} rating? <Assistant>:"

template={}
template['train']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that {} {}, what is the product {} rating? <Assistant>: {}"
template['test']="<User>: In a product graph dataset, edges connect products that are frequently purchased together. Based on the connections between products (books, music CDs, DVDs, VHS tapes), predict the average rating given by reviewers for the products. Given that a product {} that {} {}, what is the product {} rating? <Assistant>:"

class Trainer():
    def __init__(self,args):
        self.args=args
        self.num_edges=sum([eval(num_edges) for num_edges in args.num_edges.split(",")])
        self.tokenizer = self.get_tokenizer()
        self.train_loader, self.valid_loader, self.test_loader,self.data=pre_data(self.args)
        
        self.model = self.get_model()
        
        self.token_model=nn.Embedding(self.num_edges, 4096)
        self.token_model.weight.data=torch.load('edge_weights_transfer.pth')
        
        self.optimizer, self.lr_scheduler=self.get_optimizer()
        
        # if self.args.dataset == 'roman_empire':
        # self.embed_tokens = torch.cat([self.embed_tokens.cpu(), torch.zeros(self.num_edges, 4096).cpu()])
        
        
    def get_tokenizer(self):
        tokenizer = LlamaTokenizer.from_pretrained(self.args.backbone, max_length=self.args.max_text_length,do_lower_case=self.args.do_lower_case)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token
        
        if self.args.dataset=='ogbn-arxiv':
            new_tokens=[ 'node_'+str(i) for i in range(169343)]
        if self.args.dataset=='deezer-europe':
            new_tokens=[ 'node_'+str(i) for i in range(28281)]
        if self.args.dataset=='roman_empire':
            new_tokens=[ 'node_'+str(i) for i in range(22662)]
        if self.args.dataset=='amazon_ratings':
            new_tokens=[ 'node_'+str(i) for i in range(24492)]
        if self.args.dataset=='questions':
            new_tokens=[ 'node_'+str(i) for i in range(48921)]
        tokenizer.add_tokens(new_tokens)
        
        new_edges=[ 'connect_'+str(i) for i in range(1,self.num_edges+1)]
        tokenizer.add_tokens(new_edges)
        
        return tokenizer
    def get_optimizer(self):
        
        batch_per_epoch = len(self.train_loader)
        t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epoch
        warmup_ratio = self.args.warmup_ratio
        warmup_iters = int(t_total * warmup_ratio)
        
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print('Warmup ratio:', warmup_ratio)
        print("Warm up Iters: %d" % warmup_iters)
        
        
        
        for param in self.model.model.model.embed_tokens.parameters():
            param.requires_grad = True
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    'lr': self.args.lr,
                },
                # 这个组包含了bias和LayerNorm的所有参数，不应用权重衰减
                {
                    "params":[p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": 0.0,
                    'lr': self.args.lr,
                },
                {
                    "params": [p for n, p in self.token_model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
                    "weight_decay": self.args.weight_decay,
                    'lr': 1e-3,
                },

        ]
        
        optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
        lr_scheduler = get_linear_schedule_with_warmup(optim, warmup_iters, t_total)
        
        return optim, lr_scheduler
    
    def get_model(self):
        model = AutoModelForCausalLM.from_pretrained(
                                    self.args.backbone,
                                    load_in_8bit=True,
                                    torch_dtype=torch.float16,
                                    use_safetensors=False,
                                    device_map='cuda:0'
                                )
        
        # model_embed=model.model.embed_tokens.weight.data
        
        model.resize_token_embeddings(32000+self.args.num_nodes)
        
        model.model.embed_tokens.weight.data[-self.args.num_nodes:]=self.data.x
        
#         model.resize_token_embeddings(32000+self.args.num_nodes+self.num_edges)
        
#         model.model.embed_tokens.weight.data[-self.num_edges:]=0
        
        
        model = prepare_model_for_kbit_training(model)
        
        
        lora_config = LoraConfig(
            r=self.args.lora_r,
            lora_alpha=self.args.lora_alpha,
            lora_dropout=self.args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            # target_modules=['q_proj','k_proj']
            target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj']
            
        )
        
        model= get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        
        return model
    
    
    def get_prompt(self,batch,is_training=True):
        
        
        #将label又数字id形式转化为文字
        
        if self.args.dataset=='ogbn-arxiv':
            dict_labelid2categeory=load_pickle('dict_labelid2arxivcategeory.pkl')
        if self.args.dataset=='deezer-europe':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='female'
            dict_labelid2categeory[1]='male'
        if self.args.dataset=='roman_empire':
            dict_labelid2categeory={}
            
            dict_labelid2categeory = {1: 'prepositional object',2: 'preposition',3: 'determiner',4: 'adjectival',5: 
                                    'conjunct',6: 'nominal subject',7: 'coordinating conjunction',0: 'root',
                                    8: 'direct object',9: 'adverbial',10: 'compound',11: 'auxiliary',
                                    12: 'appositional',13: 'passive auxiliary',14: 'passive nominal subject',15:
                                    'possession',16: 'relative clause',17: 'other'}
            
            
        if self.args.dataset=='amazon_ratings':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='Very Positive'
            dict_labelid2categeory[1]='Positive'
            dict_labelid2categeory[2]='Neutral'
            dict_labelid2categeory[3]='Negative'
            dict_labelid2categeory[4]='Very Negative'
            
        if self.args.dataset=='questions':
            dict_labelid2categeory={}
            dict_labelid2categeory[0]='activate'
            dict_labelid2categeory[1]='no'
        
        neighbors_dict=get_first_and_second_order_neighbors(batch,batch.n_id[:batch.batch_size])
        
        batch_text=[]
        labels=[]
        
        # ' connect_'+str(len(neighbors_dict[i][j])) 
        
        for i,label in zip(neighbors_dict.keys(),batch.y[:batch.batch_size]):
            label=dict_labelid2categeory[label.item()]
            connect_text='['
            text=''
            for j in neighbors_dict[i].keys():
                connect_text+='node_'+str(j) + ' connect connect_'+str(len(neighbors_dict[i][j]))+' [ '+','.join('node_'+str(item) for item in neighbors_dict[i][j]) + '],'
            connect_text=connect_text[:-1]+']'
            if is_training :
                text = template['train'].format('node_'+str(i),'connect connect_'+str(len(neighbors_dict[i][j])) ,connect_text,'node_'+str(i),label)+'</s>'
            else:
                text = template['test'].format('node_'+str(i),'connect connect_'+str(len(neighbors_dict[i][j])) ,connect_text,'node_'+str(i))
            batch_text.append(text)
            labels.append(label+'</s>')
        input_ids=self.tokenizer(batch_text,padding='longest',
                                 max_length=self.args.max_text_length,return_tensors="pt")['input_ids']
        attention_mask=self.tokenizer(batch_text,padding='longest',
                                      max_length=self.args.max_text_length,return_tensors="pt")['attention_mask']
        
        #去掉开头的字符
        label_ids=self.tokenizer(labels,padding='longest',
                                 max_length=self.args.max_text_length,return_tensors="pt")['input_ids']
        
        
        if is_training:
            
            label_ids[label_ids.eq(self.tokenizer.pad_token_id)]=-100
            label_ids[:,-1]=2
            label_ids[label_ids.eq(1)]=-100
            label_ids=torch.cat((torch.full((label_ids.size(0), input_ids.size(-1)-label_ids.size(-1)), -100),
                              label_ids),dim=-1)
        else:
            # 测试阶段可能不需要生成标签
            label_ids = labels
        
        return input_ids, attention_mask, label_ids,neighbors_dict
    
    def load_checkpoint(self, ckpt_path,proj_path):
        results = self.model.load_state_dict(torch.load(ckpt_path), strict=True)  
        results = self.proj_model.load_state_dict(torch.load(proj_path), strict=True)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            print('Model loaded from ', proj_path)
            print(results)
    def train(self):
        
        self.model.train()
        
        self.token_model.train()
            
        self.token_model=self.token_model.to(self.args.device)
        
        pbar = tqdm(total=len(self.train_loader), ncols=275)
        for epoch in range(self.args.epoch):
            
            for step_i, batch in enumerate(self.train_loader):
                input_ids, attention_mask, labels, neighbors_dict=self.get_prompt(batch,True)
                
                attention_mask=attention_mask.to(self.args.device)
                labels=labels.to(self.args.device)
                
                embed_tokens=self.model.model.model.embed_tokens(input_ids).to(self.args.device)
                        
                input_ids=input_ids.to(self.args.device)
                
                if self.args.dataset=='ogbn-arxiv':
                    E=self.E[input_ids].to(self.args.device)
                    
                    embeds = embed_tokens+zeros_tensor
                else:
                    zeros_tensor = torch.zeros(input_ids.shape[0],input_ids.shape[1], 4096, device=input_ids.device)

                    for i in range(input_ids.shape[0]):
                        input_indices = torch.where(input_ids[i] >= (32000 + self.args.num_nodes))[0]
                        input_weights = self.token_model(input_ids[i,input_indices]-32000-self.args.num_nodes)
                        zeros_tensor[i,input_indices]=input_weights
                    
                    embeds = embed_tokens+zeros_tensor
                
                
                output= self.model(
                    inputs_embeds=embeds,
                    labels=labels,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True)
                
                loss = output['loss']/ self.args.gradient_accumulation_steps
                
                
                loss.backward()
                
                    
                if step_i % self.args.gradient_accumulation_steps == 0:
                                        # 在训练循环中调用
                    # grad_norm = get_total_grad_norm(self.model.parameters())
                    # print(f"Gradient Norm model: {grad_norm}")
                    # grad_norm = get_total_grad_norm(self.token_model.parameters())
                    # print(f"Gradient Norm token_model: {grad_norm}")
                    
                    parameters = list(self.token_model.parameters())+list(self.model.parameters())
                    torch.nn.utils.clip_grad_norm_(parameters, self.args.clip_grad_norm)
        
                    # grad_norm = get_total_grad_norm(self.model.parameters())
                    # print(f"Gradient Norm model: {grad_norm}")
                    # grad_norm = get_total_grad_norm(self.token_model.parameters())
                    # print(f"Gradient Norm token_model: {grad_norm}")
                    
                    self.optimizer.step()  # Update
                    self.lr_scheduler.step()
                    for param in self.model.parameters():
                        param.grad = None
                    for param in self.token_model.parameters():
                        param.grad = None
                    self.model.model.model.embed_tokens.weight.data[-self.num_edges:]=0
                if step_i % 1 == 0:
                    lr = self.lr_scheduler.get_lr()[0]
                    desc_str = f'Epoch {epoch} | LR {lr:.6f} |'
                    desc_str += f' Loss:{loss:.3f}'
                    pbar.set_description(desc_str)
                    pbar.update(1)
            # if epoch==2:
            #     torch.save(self.model.state_dict(),"llmcom_{}_end.pth".format(epoch+1))
            #     torch.save(self.token_model.state_dict(),"token_{}_end.pth".format(epoch+1))
                # torch.save(self.proj_model.state_dict(),"projcom_{}_end.pth".format(epoch+1))
        pbar.close()
        with torch.no_grad():   
            torch.save(self.model.state_dict(),"llmcom_{}_end.pth".format(self.args.epoch))
            torch.save(self.token_model.state_dict(),"token_{}_end.pth".format(self.args.epoch))
                                         
    def test(self):
        for epoch in range(1):
            ckpt_path = "llmcom_1_end_log1.pth"
            # token_path = "token_1_end.pth"

            # print(ckpt_path)
            # print(proj_path)
            self.model.load_state_dict(torch.load(ckpt_path), strict=True)
            # self.token_model.load_state_dict(torch.load(token_path), strict=True)
            
            self.model.eval()
            self.token_model.eval()
            self.token_model=self.token_model.to(self.args.device)
            
            self.model.model.resize_token_embeddings(32000+self.args.num_nodes+self.num_edges)
        
            self.model.model.model.embed_tokens.weight.data[-self.num_edges:]=0
            

            with torch.no_grad():
                print('len of val_loader is {}'.format(len(self.test_loader)))
                acc=0
                self.auc_hat=[]
                self.auc=[]
                for step_i, batch in tqdm(enumerate(self.test_loader)):

                    input_ids, attention_mask, labels,neighbors_dict=self.get_prompt(batch,False)
                    
                    attention_mask=attention_mask.to(self.args.device)
                    
                    input_ids=input_ids.to(self.args.device)
                    
                    # embed_tokens=self.embed_tokens[input_ids].to(self.args.device)
                    embeds=self.model.model.model.embed_tokens(input_ids).to(self.args.device)
                    
                    if self.args.dataset == 'ogbn-arxiv':

                        embeds = embed_tokens + self.token_model(input_ids)
                    else:
                        zeros_tensor = torch.zeros(input_ids.shape[0],input_ids.shape[1], 4096, device=input_ids.device)

                        for i in range(input_ids.shape[0]):
                            input_indices = torch.where(input_ids[i] >= (32000 + self.args.num_nodes))[0]
                            input_weights = self.token_model(input_ids[i,input_indices]-32000-self.args.num_nodes)
                            zeros_tensor[i,input_indices]=input_weights

                        embeds = embeds+zeros_tensor
                    
                    output= self.model.generate(inputs_embeds=embeds,
                                                attention_mask=attention_mask,max_new_tokens=20,num_beams=1)
                    output=self.tokenizer.batch_decode(output)
                    print(output)
                    print(labels)
                    for i in range(len(output)):
                        if labels[i] in output[i]:
                           acc+=1
                    print(acc)
                        
def set_random_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def main():
    args=get_args()
    seed_value = 42
    set_random_seed(seed_value)
    # trainer=Trainer(args)
    # if args.train==True:
    #     trainer.train()
    trainer=Trainer(args)
    if args.test==True:
        # 设置一个固定的随机种子值     
        trainer.test()
    
    

if __name__=='__main__':
    main()