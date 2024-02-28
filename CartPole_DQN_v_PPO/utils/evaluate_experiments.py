from utils.load_experimental_results import load_experimental_results
from utils.get_experiment_names import get_experiment_names
from utils.gen_stats import gen_stats
import numpy as np
import os
import statistics
import pandas as pd

def evaluate_experiments():

    # Set location of numpy files
    ppo_numpy_file_loc = 'ppo_experiments/numpy_files/'
    dqn_numpy_file_loc = 'dqn_experiments/numpy_files/'

    # Get all the experiment names
    ppo_experiment_names = get_experiment_names('ppo')
    dqn_experiment_names = get_experiment_names('dqn')

    # Create a dataframe to store the results
    dqn_df = []#pd.DataFrame(columns=['experiment_name', 'avg_reward', 'avg_number_of_episodes', 'avg_training_time'])
    ppo_df = []#pd.DataFrame(columns=['experiment_name', 'avg_reward', 'avg_number_of_episodes', 'avg_training_time'])

    for name in dqn_experiment_names:
        results_string = name+'_experimental_results.npy'
        num_episodes_string = name+'_num_episodes_per_experiment.npy'
        time_string = name+'_training_time_per_experiment.npy'

        results = np.load(dqn_numpy_file_loc+results_string, allow_pickle=True)
        num_episodes = np.load(dqn_numpy_file_loc+num_episodes_string)
        time = np.load(dqn_numpy_file_loc+time_string)

        # Get the stats for each experiment
        try:
            avg_reward, avg_number_of_episodes, avg_training_time = gen_stats(results, num_episodes, time)
            avg_number_of_episodes = avg_number_of_episodes[0]
            avg_avg_reward = round(statistics.mean(avg_reward),2)
        except:
            print('Error in experiment: ', name)  
            continue  

        dqn_df.append([name, avg_avg_reward, avg_number_of_episodes, avg_training_time])
        #dqn_df.append({'experiment_name': name, 'avg_reward': avg_avg_reward, 'avg_number_of_episodes': avg_number_of_episodes, 'avg_training_time': avg_training_time}, ignore_index=True)

    for name in ppo_experiment_names:
        results_string = name+'_experimental_results.npy'
        num_episodes_string = name+'_num_episodes_per_experiment.npy'
        time_string = name+'_training_time_per_experiment.npy'
        
        results = np.load(ppo_numpy_file_loc+results_string, allow_pickle=True)
        num_episodes = np.load(ppo_numpy_file_loc+num_episodes_string)
        time = np.load(ppo_numpy_file_loc+time_string)

        # Get the stats for each experiment
        try:
            avg_reward, avg_number_of_episodes, avg_training_time = gen_stats(results, num_episodes, time)
            avg_number_of_episodes = avg_number_of_episodes[0]
            avg_avg_reward = round(statistics.mean(avg_reward),2)
            avg_training_time = round(avg_training_time, 2)
        except: 
            print('Error in experiment: ', name)
            continue
        
        ppo_df.append([name, avg_avg_reward, avg_number_of_episodes, avg_training_time])
        #ppo_df.append({'experiment_name': name, 'avg_reward': avg_avg_reward, 'avg_number_of_episodes': avg_number_of_episodes, 'avg_training_time': avg_training_time}, ignore_index=True)
    
    dqn_df = np.array(dqn_df)
    ppo_df = np.array(ppo_df)

    dqn_avg_reward = dqn_df[:,1].astype(float)
    dqn_avg_number_of_episodes = dqn_df[:,2].astype(float)
    dqn_avg_training_time = dqn_df[:,3].astype(float)

    ppo_avg_reward = ppo_df[:,1].astype(float)
    ppo_avg_number_of_episodes = ppo_df[:,2].astype(float)
    ppo_avg_training_time = ppo_df[:,3].astype(float)

    best_dqn_reward = np.argmax(dqn_avg_reward)
    best_dqn_episode = np.argmin(dqn_avg_number_of_episodes)
    best_dqn_time = np.argmin(dqn_avg_training_time)   

    best_ppo_reward = np.argmax(ppo_avg_reward)
    best_ppo_episode = np.argmin(ppo_avg_number_of_episodes)
    best_ppo_time = np.argmin(ppo_avg_training_time) 

    dqn_names_best = [dqn_df[best_dqn_reward,0], dqn_df[best_dqn_episode,0], dqn_df[best_dqn_time,0]]
    ppo_names_best = [ppo_df[best_ppo_reward,0], ppo_df[best_ppo_episode,0], ppo_df[best_ppo_time,0]]

    print('Best DQN reward: ', dqn_df[best_dqn_reward,0])
    print('Best DQN episode: ', dqn_df[best_dqn_episode,0])
    print('Best DQN time: ', dqn_df[best_dqn_time,0])

    print('Best PPO reward: ', ppo_df[best_ppo_reward,0])
    print('Best PPO episode: ', ppo_df[best_ppo_episode,0])
    print('Best PPO time: ', ppo_df[best_ppo_time,0])
        
    return dqn_names_best, ppo_names_best, dqn_df, ppo_df

