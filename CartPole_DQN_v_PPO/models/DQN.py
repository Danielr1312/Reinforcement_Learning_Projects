import torch
import torch. nn as nn

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(DQN, self).__init__()

        self.state_size = state_dim
        self.action_size = action_dim
        self.hidden_size = hidden_dim
        self.l1 = nn.Linear(self.state_size, self.hidden_size)
        self.l2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.l3 = nn.Linear(self.hidden_size, self.action_size)
        self.relu = nn.ReLU()


    def forward(self, state):
        x = self.l1(state)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x