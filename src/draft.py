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

        # Master
        self.m_rnn = nn.RNNCell(5 + 2 * osc_hid, m_hid)
        self.m_out = nn.Linear(m_hid, 2 * osc_hid)
        self.m_s = torch.Variable(1, m_hid)

        # Hip
        self.h_rnn = nn.RNNCell(1 + 2 * osc_hid, osc_hid)
        self.h_out = nn.Linear(osc_hid, 1)
        self.h1_s = torch.Variable(1, osc_hid)
        self.h2_s = torch.Variable(1, osc_hid)

        # Knee
        self.k_rnn = nn.RNNCell(1 + 2 * osc_hid, osc_hid)
        self.k_out = nn.Linear(osc_hid, 1)
        self.k1_s = torch.Variable(1, osc_hid)
        self.k2_s = torch.Variable(1, osc_hid)

        # Foot
        self.f_rnn = nn.RNNCell(1 + osc_hid, osc_hid)
        self.f_out = nn.Linear(osc_hid, 1)
        self.f1_s = torch.Variable(1, osc_hid)
        self.f2_s = torch.Variable(1, osc_hid)

    def forward(self, x):
        h1, k1, f1, h2, k2, f2, *m_obs = x[0, [2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10]]

        self.m_s = self.m_rnn(self.m_s, m_obs + self.h1_s + self.h2_s)
        out_m = self.m_out(self.m_s)

        self.h1_s = self.h_rnn(self.h1_s, h1 + out_m[:, :self.osc_hid] + self.k1)
        self.h2_s = self.h_rnn(self.h2_s, h2 + out_m[:, self.osc_hid:] + self.k2)

        self.k1_s = self.k_rnn(self.k1_s, k1 + self.h1 + self.f1)
        self.k2_s = self.k_rnn(self.k2_s, k2 + self.h2 + self.f2)

        self.f1_s = self.k_rnn(self.f1_s, f1 + self.f1)
        self.f2_s = self.k_rnn(self.f2_s, f2 + self.f2)

        out_h1 = self.h_out(self.h1_s)
        out_h2 = self.h_out(self.h2_s)

        out_k1 = self.k_out(self.k1_s)
        out_k2 = self.k_out(self.k1_s)

        out_f1 = self.f_out(self.f1_s)
        out_f2 = self.f_out(self.f2_s)

        action = torch.stack([out_m, out_h1, out_k1, out_f1, out_h2, out_k2, out_f2])

        return action

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

# add the model on top of the convolutional base
model = RNet(4,2)
model.apply(weights_init)

def get_reward(weights, model, render=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    env = gym.make("Walker2d-v2")
    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            pass
            #env.render()
        batch = torch.from_numpy(ob[np.newaxis, ...]).float()
        prediction = cloned_model(Variable(batch, volatile=True))
        action = prediction.data[0]
        ob, reward, done, _ = env.step(action)

        total_reward += reward

    #env.close()
    return total_reward


partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=50,
    sigma=0.1, learning_rate=0.001, reward_goal=300, consecutive_goal_stopping=20,
    threadcount=8, cuda=False, render_test=True
)
start = time.time()
final_weights = es.run(4000, print_step=1)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath("es_walkerweights"), 'wb'))

reward = partial_func(final_weights, render=True)

print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")