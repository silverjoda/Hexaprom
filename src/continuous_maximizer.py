import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import gym
import time

class Model(nn.Module):
    def __init__(self, obs_dim, act_dim, n_hid):
        super(Model, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hid = n_hid

        # Set states
        self.reset()

        self.rnn_out = nn.GRUCell(obs_dim + act_dim, n_hid)
        self.l1 = nn.Linear(n_hid, 16)
        self.out = nn.Linear(16, obs_dim)


    def forward(self, x):
        self.h = self.rnn(x, self.h)
        l1 = self.l1(self.h)
        return self.m_out(l1)


    def reset(self, batchsize=1):
        self.h = Variable(torch.zeros(batchsize, self.n_hid)).float()


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_hid):
        super(Policy, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hid = n_hid

        # Set states
        self.reset()

        self.rnn_out = nn.GRUCell(obs_dim, n_hid)
        self.l1 = nn.Linear(n_hid, 16)
        self.out = nn.Linear(16, act_dim)


    def forward(self, x):
        self.h = self.rnn(x, self.h)
        l1 = self.l1(self.h)
        return self.m_out(l1)


    def reset(self, batchsize=1):
        self.h = Variable(torch.zeros(batchsize, self.n_hid)).float()


def main():

    # Create environment
    env = gym.make("Hopper-v2")
    obs_dim = env.observation_space
    act_dim = env.action_space

    # Create prediction model
    model = Model(obs_dim, act_dim, obs_dim)

    # Create policy model
    policy = Policy(obs_dim, act_dim, obs_dim)

    # Train prediction model on random rollouts
    model_eps = 1000
    for i in range(model_eps):
        s = env.reset()
        done = False

        episode = []
        while not done:



    # Training algorithm:

    pass

if __name__=='__main__':
    main()