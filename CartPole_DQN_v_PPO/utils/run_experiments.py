import time
import numpy as np
from utils.plot_results import plot_results
from models.QAgent import QAgent
from models.PPOAgent import PPOAgent
from utils.check_exists import check_exists

def run_experiments_fn(agent_parameters, model_type = 'dqn', run_experiments = True, num_experiments = 3):
    '''
    Description:
        Runs experiments for a given agent and model type. Saves the results in numpy files

    Argumentss:
        agent_parameters: dictionary of agent parameters
        model_type: string, either 'dqn' or 'ppo'
        run_experiments: boolean, if True, runs experiments. If False, trains a single model.

    Returns:
        experimental_results: list of lists, each list is the reward history of an experiment
        avg_rwd: list, average reward per experiment
        num_episodes_per_experiment: list, number of episodes per experiment
        training_time_per_experiment: list, training time per experiment
    '''

    if model_type == 'dqn':
        np_path = 'dqn_experiments/numpy_files/'
        plot_path = 'dqn_experiments/plots/'
    elif model_type == 'ppo':
        np_path = 'ppo_experiments/numpy_files/'
        plot_path = 'ppo_experiments/plots/'
    else:
        raise ValueError('type must be either "dqn" or "ppo"')
    
    # check if experiment has already been run
    title = agent_parameters['title']
    for key in agent_parameters:
        if key != 'title' and key != 'verbose' and key != 'train_cartpole':
            title += '_' + str(agent_parameters[key])
    if check_exists(title, model_type):
        print(f'Experiment {title} has already been run. Skipping...')
        print(f'Plot can be found at {plot_path+title+"_experimental_results.png"}')
        print(f'Experimental results can be found at {plot_path+title+"_experimental_results.npy"}')
        print(f'At the bottom of the ipynb file, use load_experimental_results() to load the results.')
        return None, None, None, None

    experimental_results = []
    num_episodes_per_experiment = []
    training_time_per_experiment = []
    if run_experiments:
        for i in range(num_experiments):
            print('---------------------------------')
            print(f"Experiment {i+1}/{num_experiments}")
            print('---------------------------------')
            if model_type == 'dqn':
                agent = QAgent(**agent_parameters)
            elif model_type == 'ppo':
                agent = PPOAgent(**agent_parameters)
            else:
                raise ValueError('type must be either "dqn" or "ppo"')
            reward_history, num_episodes, training_time, title = agent.train()
            experimental_results.append(reward_history)
            num_episodes_per_experiment.append(num_episodes)
            training_time_per_experiment.append(training_time)

        print('\n---------------------------------')
        print('Experimental results:')
        print('---------------------------------')
        avg_rwd = [sum(exp[-100:-1])/100 for exp in experimental_results]
        avg_epi = [sum(num_episodes_per_experiment)/len(num_episodes_per_experiment)]
        avg_time = sum(training_time_per_experiment)/len(training_time_per_experiment)
        print(f'\naverage reward of last 100 episodes: {avg_rwd}')
        print(f'average number of episodes to solve: {avg_epi}')
        print(f'average training time: {avg_time}')

        experimental_results_np = np.array(experimental_results, dtype=object)
        np.save(np_path+title+'_experimental_results', experimental_results_np)
        np.save(np_path+title+'_num_episodes_per_experiment', np.array(num_episodes_per_experiment))
        np.save(np_path+title+'_training_time_per_experiment', np.array(training_time_per_experiment))

        figure = plot_results(experimental_results, title+'_experimental_results', run_experiments=run_experiments, num_experiments=num_experiments)
        figure.savefig(plot_path+title+'_experimental_results.png')

        return experimental_results, avg_rwd, num_episodes_per_experiment, training_time_per_experiment
    else:
        if model_type == 'dqn':
                agent = QAgent(**agent_parameters)
        elif model_type == 'ppo':
            agent = PPOAgent(**agent_parameters)
        else:
            raise ValueError('type must be either "dqn" or "ppo"')
        reward_history, num_episodes, training_time, title = agent.train()
        avg_rwd = [sum(reward_history[-100:-1])/100]

        figure = plot_results(reward_history, title+'_experimental_results', run_experiments=run_experiments, num_experiments=num_experiments)
        figure.savefig(plot_path+title+'.png')

        return reward_history, avg_rwd, num_episodes, training_time 