import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time

# from evostra import EvolutionStrategy
from pytorch_es import EvolutionModule
from pytorch_es.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

gym_logger.setLevel(logging.CRITICAL)


class RNet(nn.Module):

    def __init__(self, m_hid, osc_hid):
        super(RNet, self).__init__()
        self.m_hid = m_hid
        self.osc_hid = osc_hid

        # Set states
        self.reset()

        # Master
        self.m_rnn = nn.GRUCell(5 + 2 * osc_hid, m_hid)
        self.m_out = nn.Linear(m_hid, 2 * osc_hid)

        # Hip
        self.h_rnn = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_out = nn.Linear(osc_hid, 1)

        # Knee
        self.k_rnn = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_out = nn.Linear(osc_hid, 1)

        # Foot
        self.f_rnn = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_out = nn.Linear(osc_hid, 1)


    def forward(self, x):

        h1 = x[:, 2]
        k1 = x[:, 3]
        f1 = x[:, 4]
        h2 = x[:, 5]
        k2 = x[:, 6]
        f2 = x[:, 7]
        m_obs = x[:, [0, 1, 8, 9, 10]]

        self.m_s = self.m_rnn(torch.cat([m_obs , self.h1_s , self.h2_s], 1), self.m_s)
        out_m = self.m_out(self.m_s)

        h1_s = self.h_rnn(torch.cat([h1.unsqueeze(0), out_m[:, :self.osc_hid], self.k1_s], 1), self.h1_s)
        h2_s = self.h_rnn(torch.cat([h2.unsqueeze(0), out_m[:, self.osc_hid:], self.k2_s], 1), self.h2_s)

        k1_s = self.k_rnn(torch.cat([k1.unsqueeze(0), self.h1_s, self.f1_s], 1), self.k1_s)
        k2_s = self.k_rnn(torch.cat([k2.unsqueeze(0), self.h2_s, self.f2_s], 1), self.k2_s)

        f1_s = self.f_rnn(torch.cat([f1.unsqueeze(0), self.f1_s], 1), self.f1_s)
        f2_s = self.f_rnn(torch.cat([f2.unsqueeze(0), self.f2_s], 1), self.f2_s)

        self.h1_s = h1_s
        self.h2_s = h2_s
        self.k1_s = k1_s
        self.k2_s = k2_s
        self.f1_s = f1_s
        self.f2_s = f2_s

        out_h1 = self.h_out(self.h1_s)
        out_h2 = self.h_out(self.h2_s)

        out_k1 = self.k_out(self.k1_s)
        out_k2 = self.k_out(self.k2_s)

        out_f1 = self.f_out(self.f1_s)
        out_f2 = self.f_out(self.f2_s)

        action = torch.cat([out_h1, out_k1, out_f1, out_h2, out_k2, out_f2], 1)

        return action

    def reset(self):
        # Master
        self.m_s = Variable(torch.zeros(1, self.m_hid)).float()

        # Hip
        self.h1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h2_s = Variable(torch.zeros(1, self.osc_hid)).float()

        # Knee
        self.k1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k2_s = Variable(torch.zeros(1, self.osc_hid)).float()

        # Foot
        self.f1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.f2_s = Variable(torch.zeros(1, self.osc_hid)).float()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train_imitation(model, trajectories, iters):
    N = len(trajectories)
    obs_dim = len(trajectories[0][0][0])
    act_dim = len(trajectories[0][0][1])

    print("Starting training. Obs dim: {}, Act dim: {}".format(obs_dim, act_dim))

    lossfun = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for i in range(iters):
        # Sample random whole episodes
        rand_episode = trajectories[np.random.randint(0, N)]
        obs_list, act_list = zip(*rand_episode)
        obs_array = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_array = torch.from_numpy(np.array(act_list, dtype=np.float32))

        y_pred_list = []

        # Rollout
        model.reset()
        optimizer.zero_grad()
        for obs in obs_array:
            y_pred_list.append(model(obs.unsqueeze(0)))
        y_preds = torch.cat(y_pred_list, 0)

        # MSE & gradients
        loss = lossfun(y_preds, act_array)
        loss.backward( )
        optimizer.step()

        print("Iteration: {}/{}, Loss: {}".format(i, iters, loss))

# add the model on top of the convolutional base
model = RNet(4,2)
model.apply(weights_init)

# Load trajectories
trajectories = pickle.load(open("/home/silverjoda/SW/baselines/data/Walker2d-v2_rollouts", 'rb'))

# Train model to imitate trajectories
train_imitation(model, trajectories, 1000)