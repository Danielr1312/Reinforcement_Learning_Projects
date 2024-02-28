import numpy as np

def gen_stats(experimental_results, num_episodes_per_experiment, training_time_per_experiment):
    avg_rwd = [sum(exp[-100:-1])/100 for exp in experimental_results]
    avg_epi = [sum(num_episodes_per_experiment)/len(num_episodes_per_experiment)]
    avg_time = sum(training_time_per_experiment)/len(training_time_per_experiment)

    return avg_rwd, avg_epi, avg_time