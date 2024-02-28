import os
import numpy as np
from utils.check_exists import check_exists

def load_experimental_results(agent_parameters_or_title, is_parameters = True, model_type = 'dqn', help = False):
    '''
    Description:
        Loads the experimental results from the numpy files saved by run_experiments.py
        and displays the plots. If the experiment has not been run, it will return None.
        Example usage can be found near the bottom of Pattern Rec Project.ipynb

    Arguments:
        agent_parameters_or_title: dictionary, parameters of the agent; or string, title of the experiment
        is_parameters: boolean, if True, agent_parameters_or_title is a dictionary of parameters, else it is a string
            note: if is_parameters is False, then the title of the experiment must exactly match the title of the experiment
            without the '_experimental_results.npy' or '_num_episodes_per_experiment.npy' or '_training_time_per_experiment.npy'
        model_type: string, either 'dqn' or 'ppo' (only necessary if is_parameters is True)
        help: boolean, if True, prints the help message

    Returns:
        experimental_results: list of lists, each list is the reward history of an experiment
        num_episodes_per_experiment: list, number of episodes per experiment
        training_time_per_experiment: list, training time per experiment
        title: string, title of the experiment
    '''

    if help:
        print(load_experimental_results.__doc__)
        return None, None, None, None

    if model_type == 'dqn':
        np_path = 'dqn_experiments/numpy_files/'
        plot_path = 'dqn_experiments/plots/'
    elif model_type == 'ppo':
        np_path = 'ppo_experiments/numpy_files/'
        plot_path = 'ppo_experiments/plots/'
    else:
        raise ValueError('model_type must be either "dqn" or "ppo"')
    
    # Generates the title of the experiment given the agent parameters
    if is_parameters:
        title = agent_parameters_or_title['title']
        for key in agent_parameters_or_title:
            if key != 'title' and key != 'verbose' and key != 'train_cartpole':
                title += '_' + str(agent_parameters_or_title[key])
    else:
        title = agent_parameters_or_title

    # Ensures the experiment has already been run
    if check_exists(title, model_type):
        experimental_results = np.load(np_path+title+'_experimental_results.npy', allow_pickle=True)
        num_episodes_per_experiment = np.load(np_path+title+'_num_episodes_per_experiment.npy', allow_pickle=True)
        training_time_per_experiment = np.load(np_path+title+'_training_time_per_experiment.npy', allow_pickle=True)
        return experimental_results, num_episodes_per_experiment, training_time_per_experiment, title
    else:
        raise ValueError('Experiment has not been run. Run run_experiments.py with those parameters first.')