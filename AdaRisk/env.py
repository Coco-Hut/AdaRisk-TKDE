import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from backbone import GIN,GAT,MLP
from mixhop import MixHopRGNN
import copy
import numpy as np
import networkx as nx

import warnings
warnings.filterwarnings('ignore')


class Environment:
    
    def __init__(self,
                 topK=0.1,
                 reward_ratio=0,
                 max_action=1,
                 f_hop=4,
                 use_adj=False,
                 is_add=True,
                 ) -> None:
        
        self.max_action=max_action
        self.topK=topK 
        self.reward_ratio=reward_ratio
        
        self.f_hop=f_hop
        self.use_adj=use_adj 
        self.is_add=is_add 
        
        self.state=None 
        self.real_graph=None 
        self.real_gt=None 
        
        self.mat=None 
        self._step_=1 
    
    def set_real_network(self,name,version,net_id,is_train=True):
        
        self.graphs=utils.load(name,version) 
        
        if is_train:
            self.real_graph=self.graphs['train'][net_id]
            self.real_gt=np.array(list(self.graphs['gt_train'][net_id].values()))
        else:
            self.real_graph=self.graphs['test'][net_id]
            self.real_gt=np.array(list(self.graphs['gt_test'][net_id].values()))
        
        
        if self.use_adj:
            self.mat=torch.as_tensor(nx.adjacency_matrix(self.real_graph).todense().A,dtype=torch.float32)
        else:
            self.mat=torch.as_tensor(utils.transition_mat(self.real_graph).T,dtype=torch.float32)
    
   
    def reset(self):
        
        #self.state=torch.tensor([(self.real_graph.nodes[node_id_num]['attr']) for node_id_num in list(self.real_graph.nodes)],dtype=torch.float32)
        
        self.state=self.state_simulator(self.real_graph)
        self.state=torch.cat([self.state,self.state[:,0].unsqueeze(1)],dim=1)
        
        self.default_probs=self.real_gt 
        self._step_=1 
        
        return self.state,self.mat
    
    def state_simulator(self,G):
        P=utils.transition_mat(G)
        x=np.array(list(G.graph['risks'].values()))
        X=[]
        for i in range(self.f_hop):
            x=np.matmul(P.T,x)
            X.append(x)
        X=np.array(X)
        X=X.T
        X=torch.as_tensor(X,dtype=torch.float32)
        attrs=torch.tensor([(G.nodes[node_id_num]['attr']) for node_id_num in list(G.nodes)],dtype=torch.float32)
        X=torch.cat([attrs[:,0].unsqueeze(1),X],dim=1)
        return X
    
    def precision_at_k(self,exist,target,topK):
        K=int(topK*len(target))
        TopK_pred=exist.argsort()[-K:][::-1] 
        TopK_target=target.argsort()[-K:][::-1]
        related_num=len(np.intersect1d(TopK_pred,TopK_target))
        pre_at_k=related_num/K
        return pre_at_k
    
    def Kendall_coef(self,exist,target,topK):
        K=int(topK*len(target))
        
        rank_pred=exist.argsort()[-K:][::-1]
        rank_target=target.argsort()[-K:][::-1]
        
        kd_coef=(4*np.sum(rank_pred==rank_target))/(K*(K-1))
        return kd_coef
    
    def reward_comp(self,last,now):
 
        diff_last=np.abs(last-self.default_probs)
        diff_now=np.abs(now-self.default_probs)
        
        diff=diff_last-diff_now
        
        ratio=np.array([0 if d<=0 else 1 for d in diff])*self.reward_ratio+(1-self.reward_ratio)*self.precision_at_k(now,self.default_probs,self.topK)
        
        avg_r=np.mean(ratio)
        
        rewards=torch.as_tensor(ratio,dtype=torch.float) 
        
        return avg_r,rewards.unsqueeze(1)
    
    def step(self,action):
        
        action=action.detach()
        
        default_last=copy.deepcopy(self.state[:,-1].numpy()) 
        
        self._step_+=1 
        
        action=(action+self.max_action)/2 
        
        if self.is_add:
            self.state[:,-1]=(1-1/self._step_)*self.state[:,-1]+(1/self._step_)*action 
            self.state[:,-1]=action
            
        default_now=copy.deepcopy(self.state[:,-1].numpy())
        
       
        avg_r,rewards=self.reward_comp(default_last,default_now)
        
        return self.state,avg_r,rewards