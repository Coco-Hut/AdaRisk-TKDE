import argparse
import warnings
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()

# RL
parser.add_argument('--topK',default=0.1,help='')
parser.add_argument('--max_action',default=1,help='')
parser.add_argument('--action_dim',default=1,help='') 
parser.add_argument('--ac_hid_dim',default=32,help='')
parser.add_argument('--cr_hid_dim',default=32,help='')
parser.add_argument('--is_map_action',default=True,help='')
parser.add_argument('--action_map',default=16,help='')
parser.add_argument('--gamma',default=0.99,help='')
parser.add_argument('--capacity',default=1000,help='')
parser.add_argument('--tau',default=0.02,help='')
parser.add_argument('--lock_episode',default=1,help='')

# GNN
parser.add_argument('--n_hop',default=4,help='')
parser.add_argument('--concat',default='atten',help='')
parser.add_argument('--encoder',default='MixHopRGNN',help='')
parser.add_argument('--src_state_dim',default=6,help='')
parser.add_argument('--hid_state_dim',default=32,help='')
parser.add_argument('--out_state_dim',default=32,help='')
parser.add_argument('--vanila_layer',default=2,help='')
parser.add_argument('--ablation',default='all',help='')

# training
parser.add_argument('--n_train_graph',default=5,help='')

parser.add_argument('--max_step',default=1400,help='')
parser.add_argument('--regular_iter',default=2,help='')

parser.add_argument('--ini_step',default=800,help='')
parser.add_argument('--delta_step',default=200,help='')

parser.add_argument('--random_steps',default=200,help='')
parser.add_argument('--update_freq',default=50,help='')
parser.add_argument('--train_freq',default=50,help='')

parser.add_argument('--actor_lr',default=2e-3,help='')
parser.add_argument('--critic_lr',default=2e-3,help='')
parser.add_argument('--random_param',default=False,help='')

parser.add_argument('--dropout',default=0.3,help='')
parser.add_argument('--alpha',default=0.2,help='')
parser.add_argument('--use_orthogonal_init',default=True,help='')
parser.add_argument('--f_hop',default=4,help='')

parser.add_argument('--reward_ratio',default=1,help='')
parser.add_argument('--use_adj',default=False,help='')
parser.add_argument('--is_add',default=True,help='')
parser.add_argument('--noise_std',default=0.2,help='')

# device & data
parser.add_argument('--dataset',default='loan',help='')
parser.add_argument('--version',default=1,help='')
parser.add_argument('--device',default='cuda',help='')

# test
parser.add_argument('--eval_max_step',default=800,help='')
parser.add_argument('--n_test_graph',default=5,help='')
parser.add_argument('--test_print_freq',default=100,help='')

args=parser.parse_args(args=[])