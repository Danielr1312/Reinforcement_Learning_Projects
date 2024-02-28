<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#authors">Authors</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>


## About The Project

This project aims to compare the performance of two Reinforcement Learning (RL) methods. The two methods to be compared are a Deep Q-Learning Network (DQN) and a Multi-layer Perceptron (MLP) with Proximal Policy Optimization (PPO). The algorithms were applied to the common [CartPole](https://www.gymlibrary.dev/environments/classic_control/cart_pole/) control problem through Open-AI Gym. Two kinds of DQNs were considered: one without replay and one with replay. The primary metric for comparison was the number of episodes required to solve CartPole where "solving" CartPole is defined as an average reward greater than 195 for the last 100 episodes. Experimental evidence shows that PPO performs better than the DQN in terms of the number of episodes required to solve CartPole and the amount of time it takes to train the model. In addition, although not quantified, the PPO model had better stability during training.

## Getting Started

### Dependencies

In order to avoid having to list out all of the dependencies needed for this repository, an environment.yml file is included. To create the environment from the .yml file, navigate to the location in your local machine where you cloned the repository using anaconda prompt command line. Then, type 'conda env create -f environment.yml' to create the virtual environment with the required dependencies. To activate the environment type 'conda activate RL_Gym'.

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/uf-eel6825-sp23/final-project-code-Danielr1312
   ```
2. Setup (and activate) your environment
  ```sh
  conda env create -f environment.yml
  ```

## Usage

There are two primary ways to interact with this project. The file 'Pattern Rec Project.ipynb' is the primary method. The file 'main.py' is used primarily for conducting experiments. Note, running main.py can take a long time since we experiment with many hyperparameters, so you'll probably want to not run it on your local machine.

### main.ipynb
To change the parameters of the models, you can edit the following parameters:

```
# Global parameters
run_experiments = True
num_experiments = 3
train_cartpole = True
verbose = True

# DQN Agent (QAgent) Parameters
## Hyperparameters
q_title = 'DQN'
q_hidden_dim = 64
q_lr = 0.001
q_gamma = 0.9
q_epsilon = 0.9
q_eps_decay = 0.99
q_replay = False # automatically set below
q_replay_size = 20
q_n_update = 10

# PPO Agent (PPOAgent) Parameters
## Hyperparameters
ppo_title = 'PPO'
ppo_hidden_dim = 24
ppo_gamma = 0.99
ppo_update_freq = 1
ppo_k_epoch = 3
ppo_lr = 0.02
ppo_lambda = 0.95
ppo_eps_clip = 0.2
ppo_v_coef = 1
ppo_ent_coef = 0.01
ppo_mem_size = 400
```

Running the ```run_experiments_fn()``` below will either run the experiment if it has not been run before, or tell you that it has been run already. If it has not been run, then it will train the model ```num_experiments``` times and save the numpy files and plot. If the parameters you entered have already been run, then run the function ```load_experimental_results()``` to load the numpy files and view the plot. If you need help running this function, then include the argument 'help=True' to view the description.

All functions have a 'help=False' argument in which you can set to true in order to better understand the function. If it is a class object, then you can use 'class_name.help()' to see the description. Note that the classes such as 'QAgent' are not directly imported into this notebook, so to view this you must import it and initialize it. 

At the end of the notebook a function called ```evaluate_experiments()``` is used. This function goes through all the numpy files and finds the best performing one for each of the following metrics: the average reward of the last 100 episodes, the average number of episodes required to solve CartPole, and the average training time. 

### main.py
In this file you can more or less just hit run. It goes through a range of hyper-parameter values and runs the experiments with those values one at a time and saves the numpy files and plots. If you would like to alter the experiments that are run, then you can alter the following in the code:

```
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
```
Note that, if the files already exist, the ```run_experiments_fn()``` will skip them.


## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


## Authors

Daniel Rodriguez - rodriguez.da@ufl.edu

Project Link: [https://github.com/Danielr1312/Reinforcement_Learning_Projects/tree/main/CartPole_DQN_v_PPO](https://github.com/Danielr1312/Reinforcement_Learning_Projects/tree/main/CartPole_DQN_v_PPO)


## Acknowledgements
* [Catia Silva](https://faculty.eng.ufl.edu/catia-silva/)


## Thank you

