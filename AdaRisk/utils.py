import torch
import torch.nn as nn
import torch.nn.functional as F
import sklearn
import networkx as nx
import pickle
import pandas as pd
import numpy as np

def long_tail_random(num,p=0.9,tau=2):
    digits=1-np.sqrt(np.random.rand(num))
    div_rand = np.random.rand(num)
    div_idx=np.where(div_rand<=p)[0] 
    digits[div_idx]=digits[div_idx]/tau
    return digits


# p和tau分别为为长尾控制系数
def graph_data(dataset='bitcoin'):
    # 读取数据集
    if dataset=='bitcoin':
        data=pd.read_csv('../dataset/bitcoin.csv')
    elif dataset=='wiki':
        data=pd.read_csv('../dataset/Wiki-Vote.txt',sep='\t')
    elif dataset=='power_grid':
        data=pd.read_csv('../dataset/power_grid.txt',sep=' ')
    elif dataset=='email':
        data=pd.read_csv('../dataset/email-Eu.txt',sep=' ')
    elif dataset=='p2p':
        data=pd.read_csv('../dataset/p2p.txt',sep=' ')
    elif dataset=='cit-HepTh':
        data=pd.read_csv('../dataset/cit-HepTh.edges',sep=' ',index_col=False)
    elif dataset=='pubmed':
        data=pd.read_csv('../dataset/pubmed.tab',sep='\t',skiprows=2,index_col=False,names=['na','edges','nan'])
        data['src']=data['edges'].map(lambda x:eval(x.split(' ')[0]))
        data['dst']=data['edges'].map(lambda x:eval(x.split(' ')[1]))
    elif dataset=='loan':
        data=pd.read_csv('../dataset/loan.csv',skiprows=1,names=['src','dst'])
        data_types_dict = {'src': int,'dst':int}
        data = data.astype(data_types_dict)
    
    edges=list(zip(data['src'].tolist(),data['dst'].tolist()))
    
    G=nx.DiGraph()
    G.add_edges_from(edges) 
    
    return G

def train_test_split_graph(G,train_ratio=0.6, random_state=None, shuffle=True):
    allNode=list(G.nodes)
    n_total=len(allNode)
    offset = int(n_total * train_ratio)
    if n_total == 0 or offset <1:
        return G,nx.DiGraph()
    if shuffle:
        allNode=sklearn.utils.shuffle(allNode,random_state=random_state)
    train_list = allNode[:offset]
    test_list = allNode[offset:]
    Largestcc_train = max(nx.weakly_connected_components(G.subgraph(train_list)),key=len)
    Largestcc_test = max(nx.weakly_connected_components(G.subgraph(test_list)),key=len)
    subGtrain=G.subgraph(Largestcc_train).copy()
    subGtest=G.subgraph(Largestcc_test).copy()
    return subGtrain,subGtest

    
def PossibleWorld(g,w):
    
    possible_default=dict(zip(list(g.nodes),np.zeros(g.number_of_nodes()))) 

    for i in range(w):
        
        default=dict(zip(list(g.nodes),np.zeros(g.number_of_nodes()))) 
        
        src_node=[] 
        
        for v in list(g.nodes):
            rv=np.random.rand()
            ps=g.graph['risks'][v]
            if rv<=ps:
                default[v]=1 
                src_node.append(v) 

        while len(src_node)!=0:
            vq=src_node.pop() 
            
            neighbors=list(g.successors(vq)) 
            if len(neighbors)!=0:
                for va_i in range(len(neighbors)):
                    
                    if default[neighbors[va_i]]==0:
                        re=np.random.rand()
                        
                        if re<=g.nodes[vq]['weights'][va_i]:
                            default[neighbors[va_i]]=1 
                            src_node.insert(0,neighbors[va_i]) 
                        else:
                            continue
        
        for v in list(possible_default.keys()):
            possible_default[v]=possible_default[v]+default[v]

        if (i+1)%2000==0:
            print('possible world {}'.format(i+1))
        
        
    for v in list(possible_default.keys()):
        possible_default[v]=possible_default[v]/w

    return possible_default


def graph_seq(G,n_ins):

    graph_cascades=[]
    for i in range(n_ins):
        new_G=nx.DiGraph(G) # 
        new_G.graph['risks']=dict(zip([i for i in list(G.nodes)],long_tail_random(len(list(G.nodes))).tolist()))
        for node_num in list(G.nodes):
            new_G.nodes[node_num]['weights']=long_tail_random(len(list(G.successors(node_num)))).tolist()
        
        for node_id in list(G.nodes):
            new_G.nodes[node_id]['attr']=[new_G.graph['risks'][node_id],
                                    len(list(new_G.predecessors(node_id))),
                                    len(list(new_G.successors(node_id))),
                                    nx.clustering(new_G,node_id)]
        
        graph_cascades.append(new_G)
        
    return graph_cascades

def save(graphset):

    with open('../casdata/{}/{}_v{}.pkl'.format(graphset['name'],graphset['name'],graphset['version']),'wb') as pkl_obj:
        pickle.dump(graphset,pkl_obj)
        
def load(name,version):

    with open('../casdata/{}/{}_v{}.pkl'.format(name,name,version),'rb') as pkl_obj:
        gr=pickle.load(pkl_obj) 
    
    return gr


def cascading_set(dataset='email',
                  train_ratio=0.6,
                  tr_ins=80,
                  te_ins=20,
                  version=0,
                  w=20000):
    
    graph=graph_data(dataset=dataset)
    
    subGtrain,subGtest=train_test_split_graph(graph,train_ratio=train_ratio,
                                              random_state=0) 
    
    train_seq=graph_seq(subGtrain,tr_ins)
    test_seq=graph_seq(subGtest,te_ins)
    
    i=1
    gt_train=[] 
    for g in train_seq:
        print('Dataset: {} -- version: {} -- train_ins: {}'.format(dataset,version,i))
        gt_train.append(PossibleWorld(g,w))
        i+=1
    
    i=1
    gt_test=[] 
    for g in test_seq:
        print('Dataset: {} -- version: {} -- test_ins: {}'.format(dataset,version,i))
        gt_test.append(PossibleWorld(g,w))
        i+=1
    
    graphset={'train':train_seq,'gt_train':gt_train,
              'test':test_seq,'gt_test':gt_test,
               'name':dataset,'version':version}
    
    save(graphset)
    



def transition_mat(G):
    
    idx=dict(zip([id for id in list(G.nodes)],[i for i in range(G.number_of_nodes())]))
    row=0
    P=[]
    for node_id in list(G.nodes):
        neighbor_id_mat=np.array([idx[k] for k in list(G.successors(node_id))])
        neigh=np.zeros(G.number_of_nodes())
        if len(neighbor_id_mat):
            neigh[neighbor_id_mat]=np.array(G.nodes[node_id]['weights'])
        neigh[row]=1 
        row+=1
        P.append(neigh)
    return np.array(P)



def save_model(model_file,model_name,m_version,
               data_name,d_version):

    with open('../agent/{}/{}_v{}_{}_v{}.pkl'.format(data_name,model_name,m_version,data_name,d_version),'wb') as pkl_obj:
        pickle.dump(model_file,pkl_obj)
        

def load_model(model_name,m_version,
               data_name,d_version):


    with open('../agent/{}/{}_v{}_{}_v{}.pkl'.format(data_name,model_name,m_version,data_name,d_version),'rb') as pkl_obj:
        model=pickle.load(pkl_obj)
    
    return model

def save_sl_log(log_file,model_name,m_version,
               data_name,d_version):
    
    with open('../sl/{}/{}_v{}_{}_v{}.pkl'.format(data_name,model_name,m_version,data_name,d_version),'wb') as pkl_obj:
        pickle.dump(log_file,pkl_obj)


def load_sl_log(model_name,m_version,
               data_name,d_version):

   
    with open('../sl/{}/{}_v{}_{}_v{}.pkl'.format(data_name,model_name,m_version,data_name,d_version),'rb') as pkl_obj:
        log=pickle.load(pkl_obj)
    
    return log
