# This code is a modification of the code from the following repository: https://github.com/RPC2/PPO


import torch
import torch.nn as nn
import torch.optim as optim
from models.PPO_MLP import MLP
import gym
import time
import numpy as np

class PPOAgent():
    '''
    Description:
        This class implements the PPO algorithm for the CartPole-v1 environment.
        Uses the MLP class to implement the policy and value networks.

    Initialization Arguments:
        title: Title of the agent. Used for saving the model and plotting the results.
        hidden_dim: Number of hidden units in the policy and value networks.
        gamma: Discount factor.
        update_freq: Number of episodes between each update of the policy and value networks.
        k_epoch: Number of epochs to train the policy and value networks.
        lr: Learning rate.
        lmbda: Lambda parameter for GAE.
        eps_clip: Epsilon parameter for clipping the policy loss.
        v_coef: Coefficient for the value loss.
        entropy_coef: Coefficient for the entropy loss.
        memory_size: Number of episodes to store in the memory.
        train_cartpole: Boolean indicating whether to train the agent on the CartPole-v1 environment.
        verbose: Boolean indicating whether to print the results of the training.

    Subroutines:
        help: Prints the description of the class.
        new_random_episode: Generates a new random episode.
        train: Trains the agent on the CartPole-v1 environment.
        update_networks: Updates the policy and value networks.
        add_memory: Adds a new episode to the memory.
        finish_path: Finishes the current episode.
    '''
    
    def __init__(self, title = 'PPO', hidden_dim = 64, gamma = 0.99, update_freq = 1, k_epoch = 3, lr = 0.02, lmbda = 0.95, 
                 eps_clip = 0.2, v_coef = 1,  entropy_coef = 0.01, memory_size = 400, train_cartpole = True, 
                 verbose = True):
        
        self.title = title
        self.n_hidden = hidden_dim
        self.gamma = gamma
        self.update_freq = update_freq
        self.k_epoch = 3
        self.learning_rate = lr
        self.lmbda = lmbda
        self.eps_clip = eps_clip
        self.v_coef = 1
        self.entropy_coef = entropy_coef
        self.memory_size = memory_size
        self.train_cartpole = train_cartpole
        self.verbose = verbose

        self.title = title + '_'+str(self.n_hidden)+'_'+str(self.gamma)+'_'+str(self.update_freq)+'_'+str(self.k_epoch)+'_'+str(self.learning_rate)+'_'+str(self.lmbda)+'_'+str(self.eps_clip)+'_'+str(self.v_coef)+'_'+str(self.entropy_coef)+'_'+str(self.memory_size)

        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            self.device = torch.device("cuda:1")
        elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.env = gym.make('CartPole-v1')
        self.action_size = self.env.action_space.n  # 2 for cartpole
        self.state_dim = self.env.observation_space.shape[0]  # 4 for cartpole
        if self.train_cartpole:
            self.policy_network = MLP(self.state_dim, self.action_size, hidden_dim=self.n_hidden).to(self.device)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.k_epoch,
                                                   gamma=0.999)
        self.loss = 0
        self.criterion = nn.MSELoss()
        self.memory = {
            'state': [], 'action': [], 'reward': [], 'next_state': [], 'action_prob': [], 'terminal': [], 'count': 0,
            'advantage': [], 'td_target': torch.FloatTensor([])
        }

    def help(self):
        print(PPOAgent.__doc__)
        return None

    def new_random_game(self):
        self.env.reset()
        action = self.env.action_space.sample()
        screen, reward, terminal, truncated, info = self.env.step(action)
        return screen, reward, action, terminal

    def train(self):
        episode_idx = 0
        step = 0
        reward_history = []
        avg_reward = []
        solved = False
        start = time.time()

        # A new episode
        while not solved:
            start_step = step
            episode_idx += 1
            episode_length = 0

            # Get initial state
            state, reward, action, done = self.new_random_game()
            current_state = state
            total_episode_reward = 1
            step_idx = 0

            # A step in an episode
            while not solved:
                step += 1
                step_idx += 1
                episode_length += 1

                # Choose action
                prob_a = self.policy_network.pi(torch.FloatTensor(current_state).to(self.device))
                action = torch.distributions.Categorical(prob_a).sample().item()

                # Act
                state, reward, done, _, _ = self.env.step(action)
                new_state = state

                reward = -1 if done else reward

                self.add_memory(current_state, action, reward/10.0, new_state, done, prob_a[action].item())

                current_state = new_state
                total_episode_reward += reward

                # stopping if the episode is too long
                if step_idx >= 4000:
                    done = True

                if done:
                    episode_length = step - start_step
                    reward_history.append(total_episode_reward)
                    avg_reward.append(sum(reward_history[-10:])/10.0)

                    self.finish_path(episode_length)

                    if (len(reward_history) > 100 and sum(reward_history[-100:-1]) / 100 >= 195) or episode_idx >= 2000:
                        solved = True

                    if self.verbose and (episode_idx == 0 or episode_idx % 100 == 0 or solved):
                        print(f'episode: {episode_idx}, total reward: {total_episode_reward}')
                        print(f"Average reward for last 100 episodes: {sum(reward_history[-100:-1])/100}")

                    self.env.reset()

                    break

            if episode_idx% self.update_freq == 0:
                for _ in range(self.k_epoch):
                    self.update_network()

        end = time.time()
        tot_time = round((end-start), 2)
        print(f"Training time: {tot_time} seconds")
        print(f'episodes to solve: {episode_idx}')

        self.env.close()

        return reward_history, episode_idx, tot_time, self.title

    def update_network(self):
        # get ratio
        state = torch.FloatTensor(np.array(self.memory['state'])).to(self.device)
        action =torch.tensor(np.array(self.memory['action'])).type(torch.int64)
        pi = self.policy_network.pi(state).to(self.device)
        advantage = torch.FloatTensor(np.array(self.memory['advantage']))#.to(self.device)
        td_target = torch.FloatTensor(np.array(self.memory['td_target'])).to(self.device)

        new_probs_a = torch.gather(pi.cpu(), 1, action)
        old_probs_a = torch.FloatTensor(np.array(self.memory['action_prob']))
        ratio = torch.exp(torch.log(new_probs_a) - torch.log(old_probs_a))

        # surrogate loss
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
        pred_v = self.policy_network.v(state).to(self.device)
        v_loss = 0.5 * (pred_v - td_target).pow(2)  # Huber loss
        entropy = torch.distributions.Categorical(pi).entropy()
        entropy = torch.tensor([[e] for e in entropy])
        self.loss = (-torch.min(surr1, surr2) + self.v_coef * v_loss.cpu() - self.entropy_coef * entropy).mean()

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    def add_memory(self, s, a, r, next_s, t, prob):
        if self.memory['count'] < self.memory_size:
            self.memory['count'] += 1
        else:
            self.memory['state'] = self.memory['state'][1:]
            self.memory['action'] = self.memory['action'][1:]
            self.memory['reward'] = self.memory['reward'][1:]
            self.memory['next_state'] = self.memory['next_state'][1:]
            self.memory['terminal'] = self.memory['terminal'][1:]
            self.memory['action_prob'] = self.memory['action_prob'][1:]
            self.memory['advantage'] = self.memory['advantage'][1:]
            self.memory['td_target'] = self.memory['td_target'][1:]

        self.memory['state'].append(s)
        self.memory['action'].append([a])
        self.memory['reward'].append([r])
        self.memory['next_state'].append(next_s)
        self.memory['terminal'].append([1 - t])
        self.memory['action_prob'].append(prob)

    def finish_path(self, length):
        state = torch.FloatTensor(np.array(self.memory['state'][-length:])).to(self.device)
        reward = torch.FloatTensor(np.array(self.memory['reward'][-length:])).to(self.device)
        next_state = torch.FloatTensor(np.array(self.memory['next_state'][-length:])).to(self.device)
        terminal = torch.FloatTensor(np.array(self.memory['terminal'][-length:])).to(self.device)

        td_target = reward + \
                    self.gamma * self.policy_network.v(next_state) * terminal
        delta = td_target - self.policy_network.v(state)
        delta = delta.cpu().detach().numpy()

        # get advantage
        advantages = []
        adv = 0.0
        for d in delta[::-1]:
            adv = self.gamma * self.lmbda * adv + d[0]
            advantages.append([adv])
        advantages.reverse()

        if self.memory['td_target'].shape == torch.Size([1, 0]):
            self.memory['td_target'] = td_target.data
        else:
            self.memory['td_target'] = torch.cat((self.memory['td_target'].cpu(), td_target.data.cpu()), dim=0)
        self.memory['advantage'] += advantages