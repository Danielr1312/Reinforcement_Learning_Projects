import numpy as np
import matplotlib.pyplot as plt
# importing custom modules
from utils.run_experiments import run_experiments_fn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

############################################################################################################
########################################## Model Parameters ################################################
############################################################################################################

# general experiments to run hidden_dim, lr, gamma
# dqn experiments to run epsilon, eps_decay, replay_size, n_update
# ppo experiments to run update_freq, k_epoch, lmbda, eps_clip, v_coef, entropy_coef, memory_size

# Global parameters
run_experiments = True
num_experiments = 3
train_cartpole = True
verbose = True

# Experiments to run for DQN and PPO
dqn_experiments = {
    'hidden_dim': [24, 48, 64, 128, 256],
    'lr': [0.001, 0.01, 0.02, 0.05],
    'gamma': [0.9, 0.99, 0.999],
    'epsilon': [0.9, 0.99, 0.999],
    'eps_decay': [0.9, 0.99, 0.999],
    'replay_size': [20, 40, 60, 80, 100],
    'n_update': [10, 20, 30, 40, 50],
}

ppo_experiments = {
    'hidden_dim': [24, 48, 64, 128, 256],
    'gamma': [0.9, 0.99, 0.999],
    'update_freq': [1, 5, 10],
    'k_epoch': [3, 5, 10],
    'lr': [0.001, 0.01, 0.02, 0.05],
    'lmbda': [0.9, 0.95, 0.99],
    'eps_clip': [0.1, 0.2, 0.3],
    'v_coef': [0.5, 1, 1.5],
    'entropy_coef': [0.01, 0.05, 0.1],
    'memory_size': [200, 400, 600, 800, 1000],
}

# # DQN Agent (QAgent) Parameters
# ## Hyperparameters
# q_title = 'DQN'
# q_hidden_dim = 64 # 24, 48, 64, 128, 256, 512
# q_lr = 0.001 # 0.001, 0.01, 0.02, 0.05 
# q_gamma = 0.9 # 0.9, 0.99, 0.999
# q_epsilon = 0.9 # 0.9, 0.99, 0.999
# q_eps_decay = 0.99 # 0.9, 0.99, 0.999
# #q_replay = False # automatically set below
# q_n_update = 10 # 10, 20, 30, 40, 50
# q_replay_size = 20 # 20, 40, 60, 80, 100


# # PPO Agent (PPOAgent) Parameters
# ## Hyperparameters
# ppo_title = 'PPO'
# ppo_hidden_dim = 24 # 24, 48, 64, 128, 256, 512
# ppo_gamma = 0.99 # 0.9, 0.99, 0.999
# ppo_update_freq = 1 # 1, 5, 10
# ppo_k_epoch = 3 # 3, 5, 10
# ppo_lr = 0.02 # 0.001, 0.01, 0.02, 0.05
# ppo_lambda = 0.95
# ppo_eps_clip = 0.2
# ppo_v_coef = 1
# ppo_ent_coef = 0.01
# ppo_mem_size = 400


for key in dqn_experiments:
    print('\n################################################')
    print('Running DQN experiments for ' + key + '...')
    print('################################################\n')
    for value in dqn_experiments[key]:
        print('\nRunning DQN experiments for ' + key + ' = ' + str(value) + '...')

        # Default DQN parameters
        q_params = {
            'title': 'DQN',
            'hidden_dim': 24,
            'lr': 0.001,
            'gamma': 0.9,
            'epsilon': 0.9,
            'eps_decay': 0.99,
            'n_update': 10,
            'replay_size': 20,
            'replay': False, # automatically set
            'train_cartpole': True,
            'verbose': True
        }

        q_params_replay = {
            'title': 'DQN',
            'hidden_dim': 24,
            'lr': 0.001,
            'gamma': 0.9,
            'epsilon': 0.9,
            'eps_decay': 0.99,
            'n_update': 10, 
            'replay_size': 20,
            'replay': True, # automatically set
            'train_cartpole': True,
            'verbose': True
        }

        q_params[key] = value
        q_params_replay[key] = value

        experimental_results, avg_rwd, avg_num_epi, avg_time = run_experiments_fn(agent_parameters=q_params, 
                                                                                  model_type = 'dqn', 
                                                                                  run_experiments = run_experiments, 
                                                                                  num_experiments=num_experiments)
        
        experimental_results, avg_rwd, avg_num_epi, avg_time = run_experiments_fn(agent_parameters=q_params_replay, 
                                                                                  model_type = 'dqn', 
                                                                                  run_experiments = run_experiments, 
                                                                                  num_experiments=num_experiments)
        

for key in ppo_experiments:
    print('\n################################################')
    print('Running PPO experiments for ' + key + '...')
    print('################################################\n')
    for value in ppo_experiments[key]:
        print('\nRunning PPO experiments for ' + key + ' = ' + str(value) + '...')

        # Default PPO parameters
        ppo_params = {
            'title': 'PPO',
            'hidden_dim': 24,
            'gamma': 0.99,
            'update_freq': 1,
            'k_epoch': 3,   
            'lr': 0.02,
            'lmbda': 0.95,
            'eps_clip': 0.2,
            'v_coef': 1,
            'entropy_coef': 0.01,
            'memory_size': 400,
            'train_cartpole': True,
            'verbose': True
        }
        ppo_params[key] = value

        try:
            experimental_results, avg_rwd, avg_num_epi, avg_time = run_experiments_fn(agent_parameters=ppo_params, 
                                                                                    model_type = 'ppo', 
                                                                                    run_experiments = run_experiments, 
                                                                                    num_experiments=num_experiments)
        except:
            print('ValueError: nans in gradient')
            continue

        

        


        
    