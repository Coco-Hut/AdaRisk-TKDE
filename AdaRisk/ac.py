import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import GIN,GAT,MLP
from mixhop import MixHopRGNN
import warnings
warnings.filterwarnings('ignore')


class policy_network(nn.Module):
    
    
    def __init__(self, state_dim, action_dim, hidden_width, max_action):
        super(policy_network, self).__init__()
        self.max_action = max_action
        self.LinearI = nn.Linear(state_dim, hidden_width)
        self.LinearII = nn.Linear(hidden_width, hidden_width)
        self.A = nn.Linear(hidden_width, action_dim)

    def forward(self, s):
        s = F.relu(self.LinearI(s))
        s = F.relu(self.LinearII(s))
        a = self.max_action * torch.tanh(self.A(s))  # [-max,max] 
        return a

class Actor(nn.Module):
    
    def __init__(self,src_state_dim=6,
                 hid_state_dim=32,
                 out_state_dim=16,
                 n_hop=5,
                 ac_hid_dim=32,
                 action_dim=1,
                 max_action=1,
                 concat='atten',
                 ablation='all',
                 encoder='MixHopRGNN',
                 vanila_layer=2,
                 dropout=0.3,
                 alpha=0.2,
                 use_orthogonal_init=False,
                 device='cpu'):
        
        super(Actor, self).__init__()
        
        self.max_action = max_action
        self.action_dim=action_dim
        self.encoder=encoder
        self.n_hop=n_hop
        self.dropout=dropout
        self.alpha=alpha
        self.device=device
        
        if self.encoder=='MixHopRGNN':
            self.GNNEncoder=MixHopRGNN(src_state_dim,
                                       out_state_dim,
                                       n_hop=self.n_hop,
                                       concat=concat,
                                       ablation=ablation,
                                       dropout=self.dropout,
                                       alpha=self.alpha,
                                       device=self.device)
        elif self.encoder=='GIN':
            self.GNNEncoder=GIN(in_features=src_state_dim,
                                hid_features=hid_state_dim,
                                out_features=out_state_dim,
                                alpha=self.alpha,
                                dropout=self.dropout,
                                layer_num=vanila_layer,
                                device=self.device)
        elif self.encoder=='GAT':
            self.GNNEncoder=GAT(in_features=src_state_dim,
                                hid_features=hid_state_dim,
                                out_features=out_state_dim,
                                alpha=self.alpha,
                                dropout=self.dropout,
                                layer_num=vanila_layer,
                                device=self.device)
        elif self.encoder=='MLP':
            self.GNNEncoder=MLP(in_size=src_state_dim,
                                hidden_size=hid_state_dim,
                                out_size=out_state_dim,
                                alpha=self.alpha,
                                dropout=self.dropout,
                                layers=vanila_layer,
                                device=self.device)
        
    
        self.action_policy=policy_network(out_state_dim,action_dim,ac_hid_dim,max_action).to(device)

    def forward(self, s,adj):
        
        if self.encoder=='MLP':
            s=self.GNNEncoder(s)
        else:
            s= self.GNNEncoder(s,adj)
        
        a=self.action_policy(s)
        
        return a

class Critic(nn.Module):  # According to (s,a), directly calculate Q(s,a)
    
    def __init__(self, state_dim, action_dim, hidden_width,is_map_action,action_map):
        super(Critic, self).__init__()
        
        self.is_map_action=is_map_action # 是否
        
        if self.is_map_action:
            self.action_dim=action_map
        else:
            self.action_dim=action_dim
            
        self.map_layer = nn.Linear(action_dim,self.action_dim)
        self.LinearI = nn.Linear(state_dim + self.action_dim, hidden_width)
        self.LinearII = nn.Linear(hidden_width, hidden_width)
        self.Q = nn.Linear(hidden_width, 1)

    def forward(self, s, a):
        
        if self.is_map_action:
            a=self.map_layer(a) 
            
        q = F.relu(self.LinearI(torch.cat([s, a], 1)))
        q = F.relu(self.LinearII(q))
        q = self.Q(q)
        
        return q