import logging
import pickle
import matplotlib.pyplot as plt
from pytorch_es.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import time

gym_logger.setLevel(logging.CRITICAL)

class Baseline(nn.Module):

    def __init__(self, obs_dim, act_dim):
        super(Baseline, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.0)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.0)
        x = self.fc3(x)
        return x


    def reset(self):
        pass


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        # Going up
        self.f_up_l1 = nn.Linear(2, 2)
        self.f_up_l2 = nn.Linear(2, 2)

        self.k_up_l1 = nn.Linear(4, 4)
        self.k_up_l2 = nn.Linear(4, 4)

        self.h_up_l1 = nn.Linear(6, 6)
        self.h_up_l2 = nn.Linear(6, 6)

        # Down
        self.m_down_l1 = nn.Linear(11, 16)
        self.m_down_l2 = nn.Linear(16, 16)
        self.m_down_l3 = nn.Linear(16, 6)

        self.h_down_l1 = nn.Linear(6, 4)
        self.h_down_l2 = nn.Linear(4, 4)

        self.k_down_l1 = nn.Linear(4, 2)
        self.k_down_l2 = nn.Linear(2, 2)

        self.h_act_l1 = nn.Linear(8, 4)
        self.h_act_l2 = nn.Linear(4, 1)
        self.k_act_l1 = nn.Linear(6, 4)
        self.k_act_l2 = nn.Linear(4, 1)
        self.f_act_l1 = nn.Linear(4, 4)
        self.f_act_l2 = nn.Linear(4, 1)

        torch.nn.init.xavier_uniform_(self.f_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.f_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.k_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.k_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.h_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.h_up_l2.weight)

        torch.nn.init.xavier_uniform_(self.m_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l3.weight)

        torch.nn.init.xavier_uniform_(self.k_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.k_down_l2.weight)
        torch.nn.init.xavier_uniform_(self.h_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.h_down_l2.weight)

        torch.nn.init.xavier_uniform_(self.h_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.h_act_l2.weight)
        torch.nn.init.xavier_uniform_(self.k_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.k_act_l2.weight)
        torch.nn.init.xavier_uniform_(self.f_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.f_act_l2.weight)


    def forward(self, x):

        h = x[:, 2].unsqueeze(1)
        k = x[:, 3].unsqueeze(1)
        f = x[:, 4].unsqueeze(1)
        hd = x[:, 8].unsqueeze(1)
        kd = x[:, 9].unsqueeze(1)
        fd = x[:, 10].unsqueeze(1)

        m_obs = x[:, [0, 1, 5, 6, 7]]

        f_up = F.relu(self.f_up_l2(F.relu(self.f_up_l1(torch.cat([f, fd], 1)))))
        k_up = F.relu(self.k_up_l2(F.relu(self.k_up_l1(torch.cat([k, kd, f_up], 1)))))
        h_up = F.relu(self.h_up_l2(F.relu(self.h_up_l1(torch.cat([h, hd, k_up], 1)))))

        m_down = F.relu(self.m_down_l3(F.relu(self.m_down_l2(F.relu(self.m_down_l1(torch.cat([m_obs, h_up], 1)))))))
        h_down = F.relu(self.h_down_l2(F.relu(self.h_down_l1(m_down))))
        k_down = F.relu(self.k_down_l2(F.relu(self.k_down_l1(h_down))))

        h_act = self.h_act_l2(F.relu(self.h_act_l1(torch.cat([h, hd, m_down], 1))))
        k_act = self.k_act_l2(F.relu(self.k_act_l1(torch.cat([k, kd, h_down], 1))))
        f_act = self.f_act_l2(F.relu(self.f_act_l1(torch.cat([f, fd, k_down], 1))))

        act = torch.cat([h_act, f_act, k_act], 1)

        return act


def evaluate(model, env, iters):
    print("Starting visual evaluation")
    score = 0
    for i in range(iters):
        obs = env.reset()
        done = False
        total_rew = 0
        with torch.no_grad():
            while not done:
                obs_t = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
                action = model(obs_t).numpy()
                obs, rew, done, _ = env.step(action + np.random.randn(3) * 0.0)
                total_rew += rew
                env.render()
            print("EV {}/{}, rew: {}".format(i,iters, total_rew))
        score += total_rew
    print("Total score: {}".format(score))


def train_imitation(model, baseline, trajectories, iters):
    if iters == 0 or iters == None:
        print("Skipping training")
        return

    N = len(trajectories)
    obs_dim = len(trajectories[0][0][0])
    act_dim = len(trajectories[0][0][1])

    print("Starting training. Obs dim: {}, Act dim: {}".format(obs_dim, act_dim))

    lossfun = nn.MSELoss()
    model_optim = torch.optim.Adam(model.parameters(), lr=7e-3)
    baseline_optim = torch.optim.Adam(baseline.parameters(), lr=3e-3)


    for i in range(iters):
        # Sample random whole episodes
        rand_episode = trajectories[np.random.randint(0, N)]

        obs_list, act_list = zip(*rand_episode)
        obs_array = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_array = torch.from_numpy(np.array(act_list, dtype=np.float32))

        model_optim.zero_grad()
        baseline_optim.zero_grad()

        model_preds = model(obs_array)
        baseline_preds = baseline(obs_array)

        # MSE & gradients
        model_loss = lossfun(model_preds, act_array)
        baseline_loss = lossfun(baseline_preds, act_array)

        model_loss.backward()
        baseline_loss.backward()

        model_optim.step()
        baseline_optim.step()

        if i % 10 == 0:
            print("Iteration: {}/{}, Rnn loss: {}, Baseline loss: {}".format(i, iters, model_loss, baseline_loss))

    torch.save(baseline, 'baseline_imit.pt')
    torch.save(model, 'rnn_imit.pt')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Baseline model
baseline = Baseline(11, 3)

# RNN
model = PNet()

# Load trajectories
trajectories = pickle.load(open("/home/silverjoda/SW/baselines/data/Hopper-v2_rollouts_0", 'rb'))

print("Model params: {}, baseline params: {}".format(count_parameters(model), count_parameters(baseline)))

train_imitation(model, baseline, trajectories, 3000)

env = gym.make("Hopper-v2")

print("Evaluating baseline")
baseline = torch.load('baseline_imit.pt')
evaluate(baseline, env, 30)
time.sleep(0.5)
print("Evaluating Model")
model = torch.load('rnn_imit.pt')
evaluate(model, env, 30)
