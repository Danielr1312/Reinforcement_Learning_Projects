import os 

def check_exists(title, model_type = 'dqn', help = False):
    '''
    Description:
        Checks if the experiment has already been run and returns True if it has.

    Arguments:
        title: string, title of the experiment
        model_type: string, either 'dqn' or 'ppo'
        help: boolean, if True, prints the help message

    Returns:
        boolean, True if the experiment has already been run, False otherwise
    '''

    if help:
        print(check_exists.__doc__)
        return None

    if model_type == 'dqn':
        plot_path = 'dqn_experiments/plots/'
    elif model_type == 'ppo':
        plot_path = 'ppo_experiments/plots/'
    else:
        raise ValueError('model_type must be either "dqn" or "ppo"')
    
    if os.path.exists(plot_path+title+'_experimental_results.png'):
        return True
    else:
        return False
    
