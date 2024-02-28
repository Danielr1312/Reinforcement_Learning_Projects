# This code is from the following repository: https://github.com/RPC2/PPO

import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, state_dim, action_size, hidden_dim = 64):
        super(MLP, self).__init__()
        self.action_size = action_size
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.state_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3_pi = nn.Linear(self.hidden_dim, self.action_size)
        self.fc3_v = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def pi(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_pi(x)
        return self.softmax(x)

    def v(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3_v(x)
        return x