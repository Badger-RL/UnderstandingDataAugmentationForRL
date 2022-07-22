import torch
from torch import nn

class QModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super().__init__()

        self.model_kwargs = {
            'state_dim': state_dim,
            'action_dim': action_dim,
            'hidden_size': hidden_size,
        }

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        return self.q(torch.cat([state, action], 1))