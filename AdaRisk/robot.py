import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import GIN,GAT,MLP
from mixhop import MixHopRGNN
from ac import Actor,Critic,policy_network
import copy
import random
import pickle


import warnings
warnings.filterwarnings('ignore')


class Agent(object):

    def __init__(self,
                 src_state_dim=6,
                 hid_state_dim=32,
                 out_state_dim=16,
                 n_hop=4,
                 action_dim=1,
                 ac_hid_dim=32,
                 cr_hid_dim=32,
                 is_map_action=True,
                 action_map=16,
                 max_action=1,
                 encoder='MixHopRGNN',
                 concat='atten',
                 ablation='all',
                 vanila_layer=2,
                 capacity=1000,
                 gamma=0.99,
                 tau=0.02,
                 actor_lr=3e-2,
                 critic_lr=3e-2,
                 dropout=0.4,
                 alpha=0.2,
                 use_orthogonal_init=True,
                 device='cpu'
                 ):

        self.out_state_dim=out_state_dim
        self.action_dim=action_dim
        self.is_map_action=is_map_action
        self.action_map=action_map
        self.ac_hid_dim=ac_hid_dim
        self.cr_hid_dim=cr_hid_dim
        self.max_action=max_action
        
        
        self.encoder=encoder
        
        self.capacity=capacity
        self.gamma=gamma
        self.tau=tau
        self.actor_lr=actor_lr
        self.critic_lr=critic_lr
        self.device=device

        
        self.actor=Actor(src_state_dim=src_state_dim,
                         hid_state_dim=hid_state_dim,
                         out_state_dim=out_state_dim,
                         n_hop=n_hop,
                         ac_hid_dim=ac_hid_dim,
                         max_action=max_action,
                         action_dim=action_dim,
                         concat=concat,
                         ablation=ablation,
                         vanila_layer=vanila_layer,
                         encoder=encoder,
                         dropout=dropout,
                         alpha=alpha,
                         use_orthogonal_init=use_orthogonal_init,
                         device=device
                         )
        
       
        self.actor_target = copy.deepcopy(self.actor.action_policy).to(device)
        
        
        self.critic=Critic(out_state_dim,action_dim,cr_hid_dim,is_map_action=is_map_action,action_map=action_map).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr,eps=1e-5)
        
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr,eps=1e-5)
        
        
        self.replay_buffer=[]

        
        self.adj=None 
        
    
    def modify_adj(self,adj):
        self.adj=adj.to(self.device)
    
    
    def perform_action(self,state):
        
        with torch.no_grad():
            action=self.actor(state,self.adj)
            
        return action 
    
    def clear_buffer(self):
        self.replay_buffer=[]
    
    def store_transition(self,*transition):
        if len(self.replay_buffer)==self.capacity:
            self.replay_buffer.pop(0) 
        self.replay_buffer.append(transition) 
    
    def lock_param(self,is_lock=False):
        if is_lock:
            for param in self.actor.GNNEncoder.parameters():
                param.requires_grad = False
        else:
            for param in self.actor.GNNEncoder.parameters():
                param.requires_grad = True
    
    def update_mlp(self):
        
        self.actor.action_policy=policy_network(self.out_state_dim,self.action_dim,
                                                self.ac_hid_dim,self.max_action).to(self.device)
        
        self.actor_target = copy.deepcopy(self.actor.action_policy).to(self.device)
        
        self.critic=Critic(self.out_state_dim,self.action_dim,self.cr_hid_dim,
                           is_map_action=self.is_map_action,action_map=self.action_map).to(self.device)
        
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
    
    def learning(self):
        
        state_sample=random.sample(self.replay_buffer,1)
        s0,a0,r1,s1=state_sample[0] 
        
        s0=s0.to(self.device)
        a0=a0.to(self.device)
        r1=r1.to(self.device)
        s1=s1.to(self.device)
        
        with torch.no_grad():
            if self.encoder=='MLP':
                _s0_=self.actor.GNNEncoder(s0)
            else:
                _s0_=self.actor.GNNEncoder(s0,self.adj)
        
        def critic_learning():
            
            with torch.no_grad():
                if self.encoder=='MLP':
                    _s1_=self.actor.GNNEncoder(s1)
                else:
                    _s1_=self.actor.GNNEncoder(s1,self.adj)
                    
                Q_= self.critic_target(_s1_,self.actor_target(_s1_))
                target_Q = r1+self.gamma*Q_
            
            current_Q=self.critic(_s0_,a0) 
            
            td_error=F.mse_loss(current_Q,target_Q)

            self.critic_optimizer.zero_grad()
            td_error.backward()
            self.critic_optimizer.step()
        
        def actor_learning():
            
            loss=-1*self.critic(_s0_,self.actor(s0,self.adj)).mean() 
            
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step() 
        
        
        def soft_update(net_target,net,tau):
            
            for target_param,parm in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data*(1.0-self.tau)+parm.data*self.tau)
                
        
        critic_learning()
        
        for params in self.critic.parameters():
            params.requires_grad = False
        
        
        actor_learning()
        
        
        for params in self.critic.parameters():
            params.requires_grad = True
        
        
        soft_update(self.critic_target,self.critic,self.tau)
        soft_update(self.actor_target,self.actor.action_policy,self.tau)