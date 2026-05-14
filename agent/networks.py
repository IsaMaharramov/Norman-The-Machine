import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        # LSTM to handle temporal physics (Iodine/Xenon decay)
        self.lstm = nn.LSTM(state_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output mean and log_std for the SAC Gaussian Policy
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, hidden_state=None):
        # state shape: (batch, sequence_len, state_dim)
        x, lstm_hidden = self.lstm(state, hidden_state)
        x = F.relu(self.fc1(x[:, -1, :]))
        
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std, lstm_hidden

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        # Twin-Q Architecture to prevent overestimation bias
        self.lstm = nn.LSTM(state_dim + action_dim, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.q_out = nn.Linear(hidden_dim, 1)

    def forward(self, state, action, hidden_state=None):
        # state & action -> the Q-function
        x = torch.cat([state, action], dim=-1)
        x, lstm_hidden = self.lstm(x, hidden_state)
        x = F.relu(self.fc1(x[:, -1, :]))
        return self.q_out(x), lstm_hidden