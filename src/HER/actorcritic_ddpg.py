"""
Definitions for Actor and Critic
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    """Defines actor network"""
    def __init__(self, env):
        super(Actor, self).__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
                                    
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, self.act_dim)


    def forward(self, x):
        if x.shape[0] == 1:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))

        return x
        

class Critic(nn.Module):
    """Defines critic network"""
    def __init__(self, env):
        super(Critic, self).__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.obs_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128 + self.act_dim, 128)
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 1)
        
    def forward(self, s, a):
        x = F.relu(self.bn1(self.fc1(s)))
        x = F.relu(self.fc2(torch.cat([x, a], dim=1)))
        Qval = self.fc3(x)
        return Qval
        

