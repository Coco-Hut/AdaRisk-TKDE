import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import GINLayer

class MixHopRGNN(nn.Module):
    
    def __init__(self,in_feats,
                 hid_feats,
                 n_hop,
                 dropout=0.3,
                 alpha=0.2,
                 ablation='all',
                 concat='atten',
                 device='cpu') -> None:
        super(MixHopRGNN,self).__init__()
        
        self.ablation=ablation
        self.n_hop=n_hop
        self.dropout=dropout
        self.alpha=alpha
        self.in_feats=in_feats
        self.hid_feats=hid_feats
        self.concat=concat
        self.device=device
        
        self.linear=nn.Linear(in_feats,hid_feats).to(device) 
        self.dense=nn.Linear(hid_feats*self.n_hop,self.hid_feats).to(device)
        
        self.rnn_modules=[nn.LSTMCell(hid_feats,hid_feats).to(device) for _ in range(self.n_hop+1)]
        
        self.GNNLayer=[GINLayer(hid_feats,hid_feats,device=self.device) for _ in range(self.n_hop)]
        
        # attention
        self.a=nn.Parameter(torch.zeros(size=(2*hid_feats,1))).to(device)
        nn.init.xavier_normal_(self.a.data,gain=1.414)
        
    def forward(self,X,adj):
        
        self.aggregation=[] 
        
        if self.ablation=='all':
            h,c=self.rnn_modules[0](self.linear(X)) 
        elif self.ablation=='no-gru':
            h=self.linear(X)
            
        h0=h 
        
        if self.ablation=='all':
            for i in range(self.n_hop):
                h_gnn=self.GNNLayer[i](h,adj)
                h,c=self.rnn_modules[i+1](h_gnn,(h,c))
                self.aggregation.append(h)
        elif self.ablation=='no-gru':
            for i in range(self.n_hop):
                h=self.GNNLayer[i](h,adj)
                self.aggregation.append(h)
        
        # model hop preference
        if self.concat=='cat':
            self.aggregation=torch.cat(self.aggregation,dim=1) 
            node_embedding=self.dense(self.aggregation)
            
        elif self.concat=='atten':
            
            attention=torch.cat([torch.matmul(torch.cat([self.aggregation[i],h0],dim=1),self.a) for i in range(self.n_hop)],dim=1)
            attention=F.softmax(attention,dim=1)  # [N,n_hop]
            
            node_embedding=attention[:,0].unsqueeze(1).repeat(1,self.hid_feats)*self.aggregation[0]
            for i in range(1,self.n_hop):
                node_embedding+=attention[:,i].unsqueeze(1).repeat(1,self.hid_feats)*self.aggregation[i]
                
        elif self.concat=='vanila':
            node_embedding=self.aggregation[-1] 
        
        return node_embedding
