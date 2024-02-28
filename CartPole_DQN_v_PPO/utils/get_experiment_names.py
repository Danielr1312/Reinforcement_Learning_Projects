import os

def get_experiment_names(model, help = False):
    '''
    Description:
        Goes through the specified path and returns a list of the experiment titles.

    Arguments:
        model: string, 'ppo' or 'dqn'

    Returns:
        list, list of all experiment titles in the specified path
    '''
    if help:
        print(get_experiment_names.__doc__)

    if model == 'ppo':
        path = 'ppo_experiments/plots/'
    elif model == 'dqn':
        path = 'dqn_experiments/plots/'
    else:
        raise ValueError('model must be either "ppo" or "dqn"')

    filenames = []
    for filename in os.listdir(path):
        idx = filename.rfind('_', 0, filename.rfind('_')-1)
        if idx != -1:
            filename = filename[:idx]
        filenames.append(filename)
    return filenames
