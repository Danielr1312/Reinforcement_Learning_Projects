from models.DQN import DQN
import time
import gym
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from random import sample



class QAgent():
    '''
    Description: 
        QAgent is a class that implements a Deep Q Network (DQN) to solve the CartPole-v1 environment.
        Uses the DQN class to implement the DQN algorithm.

    Arguments:
        title: Title of the agent. Used for saving the model and plotting the results.
        hidden_dim: Number of hidden units in the Q network.
        lr: Learning rate.
        gamma: Discount factor.
        epsilon: Epsilon parameter for the epsilon-greedy policy.
        eps_decay: Decay rate for the epsilon parameter.
        n_update: Number of episodes between each update of the target network.
        replay_size: Number of episodes to store in the replay memory.
        replay: Boolean indicating whether to use the replay memory.
        train_cartpole: Boolean indicating whether to train the agent on the CartPole-v1 environment.
        verbose: Boolean indicating whether to print the results of the training.
    '''

    def __init__(self, title = 'DQN', hidden_dim = 64, lr = 0.001, gamma = 0.9, epsilon = 0.9, eps_decay = 0.99,
                    n_update = 10, replay_size = 20, replay = False,  
                     train_cartpole = True, verbose = True
                     ):

        # Initialize parameters
        self.n_hidden = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.use_replay = replay
        self.replay_size = replay_size
        self.title = title
        self.n_update = n_update # not used here, but in theory this would control how often the target network is updated
        self.train_cartpole = train_cartpole
        self.verbose = verbose
        self.title = title + '_' + str(self.n_hidden) + '_' + str(self.lr) + '_' + str(self.gamma) + '_' + str(self.epsilon) + '_' + str(self.eps_decay) + '_' + str(self.n_update) + '_' +str(self.replay_size) + '_' + str(self.use_replay)  

        self.env = gym.make('CartPole-v1')
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = torch.device("cuda:1")
        elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu") 
        self.action_size = self.env.action_space.n
        self.observation_size = self.env.observation_space.shape[0]
        if self.train_cartpole:
            self.model = DQN(self.observation_size, self.action_size, self.n_hidden).to(self.device)
            self.criterion = torch.nn.MSELoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss = 0

        self.memory = []

    def train(self):
            
        final = []
        episode_idx = 1
        total_replay_time = 0
        start = time.time()
        solved = False

        # Episode loop
        while not solved:

            # reset the environment
            state, _ = self.env.reset()
            done = False
            total = 0
            step_idx = 1

            # Step loop
            while not solved:
                # Epsilon Greedy Strategy
                if random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    with torch.no_grad(): # new
                        q_values = self.model.forward(torch.Tensor(state).to(self.device))
                    action = torch.argmax(q_values).item()

                next_state, reward, done, _, _ = self.env.step(action)
                
                total += reward
                self.memory.append((state, action, next_state, reward, done))
                with torch.no_grad(): # new
                    q_values = self.model.forward(torch.Tensor(state).to(self.device)).tolist()

                # stopping if the pole is balanced for more than 4000 steps within an episode
                if step_idx % 4000 == 0:
                    done = True

                if done:
                    if not self.use_replay:
                        q_values[action] = reward
                        # new
                        y_pred = self.model.forward(torch.Tensor(state).to(self.device))
                        loss = self.criterion(y_pred, torch.autograd.Variable(torch.Tensor(q_values).to(self.device)))
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        #self.model.update(state, q_values)

                    if (len(final[:]) > 100 and sum(final[-100:-1]) / 100 >= 195) or episode_idx >= 2000:
                            solved = True
                            if episode_idx >=2000:
                                print('Training halted at 2000 episodes because CartPole was not solved')
                    
                    break

                if self.use_replay and len(self.memory) > self.replay_size:
                    t0 = time.time()
                    states, all_q_values = self.replay()
                    y_preds = self.model.forward(torch.Tensor(states).to(self.device))
                    loss = self.criterion(y_preds, torch.autograd.Variable(torch.Tensor(all_q_values).to(self.device)))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    t1 = time.time()
                    total_replay_time += (t1-t0)
                else:
                    with torch.no_grad(): # new
                        q_values_next = self.model.forward(torch.Tensor(next_state).to(self.device))
                    q_values[action] = reward + self.gamma*torch.max(q_values_next).item()
                    y_pred = self.model.forward(torch.Tensor(state).to(self.device))
                    loss = self.criterion(y_pred, torch.autograd.Variable(torch.Tensor(q_values).to(self.device)))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                state = next_state

            self.epsilon = max(self.epsilon*self.eps_decay, 0.01)
            final.append(total)

            if self.verbose and (episode_idx % 100 == 0 or solved):
                print(f'Episode: {episode_idx}, Total Reward: {total}, Epsilon: {round(self.epsilon,3)}')
                print(f'Avg Reward for last 100 episodes: {sum(final[-100:-1])/100}')
                if self.use_replay:
                    print(f'Replay Time: {round(total_replay_time,2)} seconds')
                
            episode_idx += 1

        end = time.time()
        print(f'Training Time: {round(end-start,2)} seconds')
        print(f'Total episodes to solve: {episode_idx}')

        return final, episode_idx, round(end-start,2), self.title
    
    def replay(self):
        batch = sample(self.memory, self.replay_size)
        batch_t = list(map(list, zip(*batch)))
        states = np.array(batch_t[0])
        actions = np.array(batch_t[1])
        next_states = np.array(batch_t[2])
        rewards = np.array(batch_t[3])
        dones = np.array(batch_t[4])

        states = torch.Tensor(states).to(self.device)
        actions_tensor = torch.Tensor(actions).to(self.device)
        next_states = torch.Tensor(next_states).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        dones_tensor = torch.Tensor(dones).to(self.device)

        dones_idxs = torch.where(dones_tensor == True)[0]

        #with torch.no_grad(): # this might be wrong
        all_q_vals = self.model.forward(states)
        all_q_vals_next = self.model.forward(next_states)
        #update
        all_q_vals[range(len(all_q_vals)),actions]=rewards+self.gamma*torch.max(all_q_vals_next, axis=1).values
        all_q_vals[dones_idxs.tolist(), actions_tensor[dones].tolist()]=rewards[dones_idxs.tolist()]

        return states.tolist(), all_q_vals.tolist()