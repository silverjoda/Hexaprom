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
        self.obs_dim = env.obs_dim + env.goal_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
                                    
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, self.act_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        #x = self.bn1(x)
        x = F.relu(self.fc2(x))
        #x = self.bn2(x)
        x = F.relu(self.fc3(x))
        act = torch.tanh(x)
        return act
        

class Critic(nn.Module):
    """Defines critic network"""
    def __init__(self, env):
        super(Critic, self).__init__()
        self.obs_dim = env.obs_dim + env.goal_dim
        self.act_dim = env.act_dim

        self.fc1 = nn.Linear(self.obs_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.fc2 = nn.Linear(64 + self.act_dim, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.fc3 = nn.Linear(64, 1)
        
    def forward(self, s, a):
        x = F.relu(self.fc1(s))
        #x = self.bn1(x)
        x = F.relu(self.fc2(torch.cat([x, a], dim=1)))
        Qval = self.fc3(x)
        return Qval
        

