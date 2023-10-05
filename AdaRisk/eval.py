import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from config import args
from env import Environment
import copy

import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    
    kwargs={
        'topK':args.topK,
        'reward_ratio':args.reward_ratio,
        'max_action':args.max_action,
        'f_hop':args.f_hop,
        'use_adj':args.use_adj,
        'is_add':args.is_add,   
    }
    
    env=Environment(**kwargs) 
    
    agent=utils.load_model('mixhop',1,'wiki',2) 

    default_prob=[]
    

    for episode in range(args.n_test_graph):
        
        env.set_real_network('wiki',
                        2,
                        net_id=episode,
                        is_train=False
                        ) 
        
        agent.modify_adj(env.mat) 
        agent.clear_buffer() 
        
        s0,adj=env.reset() 
        
        for step in range(args.eval_max_step):
            
            a0=agent.perform_action(s0.to('cuda'))
            s1,avg_r,r1= env.step(a0.flatten().cpu())
        
            s0=s1

            if (step+1)%args.test_print_freq==0:
                risk_prob=copy.deepcopy(env.state[:,-1].numpy())
                target_prob=env.default_probs
                pre_at_k=env.precision_at_k(risk_prob,target_prob,args.topK)
                print('Test Graph {} - step {} Pre@10{}'.format(episode+1,step+1,pre_at_k))
        
       
        default_prob.append(copy.deepcopy(env.state[:,-1].numpy()))
    
   
    utils.save_res_file(default_prob,'mixhop',1,'bitcoin',2)
