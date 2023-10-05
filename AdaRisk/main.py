#-*- coding:UTF-8 -*-
import torch
import utils
from config import args
from robot import Agent
from env import Environment
import copy
import numpy as np

import warnings
warnings.filterwarnings('ignore')

if __name__ =='__main__':
    
    kwargs={
        'topK':args.topK,
        'reward_ratio':args.reward_ratio,
        'f_hop':args.f_hop,
        'use_adj':args.use_adj,
        'is_add':args.is_add,  
    }

    params={
        'src_state_dim':args.src_state_dim,
        'hid_state_dim':args.hid_state_dim,
        'out_state_dim':args.out_state_dim,
        'n_hop':args.n_hop,
        'ac_hid_dim':args.ac_hid_dim,
        'cr_hid_dim':args.cr_hid_dim,
        'is_map_action':args.is_map_action,
        'action_map':args.action_map,
        'max_action':args.max_action,
        'action_dim':args.action_dim,
        'encoder':args.encoder,
        'concat':args.concat,
        'ablation':args.ablation,
        'vanila_layer':args.vanila_layer,
        'capacity':args.capacity,
        'gamma':args.gamma,
        'tau':args.tau,
        'actor_lr':args.actor_lr,
        'critic_lr':args.critic_lr,
        'dropout':args.dropout,
        'alpha':args.alpha,
        'use_orthogonal_init':args.use_orthogonal_init,
        'device':args.device
    }
    
    env=Environment(**kwargs) 
    agent=Agent(**params)
    

    rewards=[]
    pres=[]

    print('Training Data{}'.format(args.dataset))
    
    if args.random_param:
        print('lock the parameters')
        agent.lock_param(True)
    
    for _iter in range(1,args.regular_iter+1):
        
        print('Iteration: {}'.format(_iter))
        
        if _iter==args.regular_iter:
            
            max_step=args.max_step
            
            '''
            print('lock the parameters')
            agent.lock_param(True) # 
            agent.update_mlp() #
            '''
            
        else:
            max_step=args.ini_step+_iter*args.delta_step
        
    
        for g_id in range(args.n_train_graph):
            
            env.set_real_network(args.dataset,
                            args.version,
                            net_id=g_id,
                            is_train=True
                            ) 
            
            agent.modify_adj(env.mat) 
            agent.clear_buffer() 
            
            episode_reward=0
            
            s0,adj=env.reset() 
            
            for step in range(max_step):
                
                a0=agent.perform_action(s0.to('cuda'))
                a0 = (a0.cpu() + torch.tensor(np.random.normal(0, args.noise_std, size=a0.size()),dtype=torch.float32)).clip(-args.max_action, args.max_action)
                s1,avg_r,r1= env.step(a0.flatten())
                agent.store_transition(s0,a0,r1,s1)
                
                s0=s1
                
                episode_reward+=avg_r
                
                if step >= args.random_steps and step % args.update_freq == 0:
                    for _ in range(args.train_freq):
                        agent.learning()

                if (step+1)%args.update_freq==0:
                    print('G_id {} - step {} avg reward {},acc reward {}'.format(g_id+1,step+1,avg_r,episode_reward/(step+1)))
                    risk_prob=copy.deepcopy(env.state[:,-1].numpy())
                    target_prob=env.default_probs
                    pre_at_k=env.precision_at_k(risk_prob,target_prob,args.topK)
                    print('G_id {} - step {} Pre@10 {}\n'.format(g_id+1,step+1,pre_at_k))
                    

    utils.save_model(agent,'mixhop',8,args.dataset,args.version)
    
    