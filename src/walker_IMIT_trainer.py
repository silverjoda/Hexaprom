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

        self.fc1 = nn.Linear(obs_dim, 96)
        self.fc2 = nn.Linear(96, 64)
        self.fc3 = nn.Linear(64, act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.0)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.0)
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reset(self):
        pass


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
        self.h_rnn1 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_rnn2 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_out1 = nn.Linear(osc_hid, 1)
        self.h_out2 = nn.Linear(osc_hid, 1)

        # Knee
        self.k_rnn1 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_rnn2 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_out1 = nn.Linear(osc_hid, 1)
        self.k_out2 = nn.Linear(osc_hid, 1)

        # Foot
        self.f_rnn1 = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_rnn2 = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_out1 = nn.Linear(osc_hid, 1)
        self.f_out2 = nn.Linear(osc_hid, 1)

    def normalize(self, tensor):
        tensor_norm = torch.norm(tensor, p=2, dim=1).detach()
        return tensor.div(tensor_norm.expand_as(tensor))

    def identity(self, tensor):
        return tensor

    def forward(self, x):
        #h1, k1, f1, h2, k2, f2, *m_obs = x[0, [2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10]]

        h1 = x[:, 2]
        k1 = x[:, 3]
        f1 = x[:, 4]
        h2 = x[:, 5]
        k2 = x[:, 6]
        f2 = x[:, 7]
        m_obs = x[:, [0, 1, 8, 9, 10]]

        self.m_s = self.m_rnn(self.normalize(torch.cat([m_obs , self.h1_s , self.h2_s], 1)), self.m_s)
        out_m = self.m_out(self.m_s)

        h1_s = self.h_rnn1(self.normalize(torch.cat([h1.unsqueeze(0), out_m[:, :self.osc_hid], self.k1_s], 1)), self.h1_s)
        h2_s = self.h_rnn2(self.normalize(torch.cat([h2.unsqueeze(0), out_m[:, self.osc_hid:], self.k2_s], 1)), self.h2_s)

        k1_s = self.k_rnn1(self.normalize(torch.cat([k1.unsqueeze(0), self.h1_s, self.f1_s], 1)), self.k1_s)
        k2_s = self.k_rnn2(self.normalize(torch.cat([k2.unsqueeze(0), self.h2_s, self.f2_s], 1)), self.k2_s)

        f1_s = self.f_rnn1(self.normalize(torch.cat([f1.unsqueeze(0), self.k1_s], 1)), self.f1_s)
        f2_s = self.f_rnn2(self.normalize(torch.cat([f2.unsqueeze(0), self.k2_s], 1)), self.f2_s)

        self.h1_s = self.identity(h1_s)
        self.h2_s = self.identity(h2_s)
        self.k1_s = self.identity(k1_s)
        self.k2_s = self.identity(k2_s)
        self.f1_s = self.identity(f1_s)
        self.f2_s = self.identity(f2_s)

        out_h1 = self.h_out1(self.h1_s)
        out_h2 = self.h_out2(self.h2_s)

        out_k1 = self.k_out1(self.k1_s)
        out_k2 = self.k_out2(self.k2_s)

        out_f1 = self.f_out1(self.f1_s)
        out_f2 = self.f_out2(self.f2_s)

        action = torch.cat([out_h1, out_k1, out_f1, out_h2, out_k2, out_f2], 1)

        #print([self.m_s.data, self.h1_s.data, self.h2_s.data, self.k1_s.data, self.k2_s.data, self.f1_s.data, self.f2_s.data])

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


class RNet2(nn.Module):
    def __init__(self, m_hid, osc_hid):
        super(RNet2, self).__init__()
        self.m_hid = m_hid
        self.osc_hid = osc_hid

        # Set states
        self.reset()

        # Master
        self.m_rnn = nn.GRUCell(5 + 2 * osc_hid, m_hid)
        self.m_out = nn.Linear(m_hid, 2 * osc_hid)

        # Hip
        self.h_rnn1 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_rnn2 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_out1 = nn.Linear(osc_hid, 1)
        self.h_out2 = nn.Linear(osc_hid, 1)
        self.h_out1_down = nn.Linear(osc_hid, osc_hid)
        self.h_out2_down = nn.Linear(osc_hid, osc_hid)
        self.h_out1_up = nn.Linear(osc_hid, osc_hid)
        self.h_out2_up = nn.Linear(osc_hid, osc_hid)

        # Knee
        self.k_rnn1 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_rnn2 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_out1 = nn.Linear(osc_hid, 1)
        self.k_out2 = nn.Linear(osc_hid, 1)
        self.k_out1_down = nn.Linear(osc_hid, osc_hid)
        self.k_out2_down = nn.Linear(osc_hid, osc_hid)
        self.k_out1_up = nn.Linear(osc_hid, osc_hid)
        self.k_out2_up = nn.Linear(osc_hid, osc_hid)

        # Foot
        self.f_rnn1 = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_rnn2 = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_out1 = nn.Linear(osc_hid, 1)
        self.f_out2 = nn.Linear(osc_hid, 1)
        self.f_out1_up = nn.Linear(osc_hid, osc_hid)
        self.f_out2_up = nn.Linear(osc_hid, osc_hid)

    def normalize(self, tensor):
        tensor_norm = torch.norm(tensor, p=2, dim=1).detach()
        return tensor.div(tensor_norm.expand_as(tensor))

    def identity(self, tensor):
        return tensor

    def forward(self, x):
        #h1, k1, f1, h2, k2, f2, *m_obs = x[0, [2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10]]

        h1 = x[:, 2]
        k1 = x[:, 3]
        f1 = x[:, 4]
        h2 = x[:, 5]
        k2 = x[:, 6]
        f2 = x[:, 7]
        m_obs = x[:, [0, 1, 8, 9, 10]]

        self.m_s = self.m_rnn(self.identity(torch.cat([m_obs , self.h1_up , self.h2_up], 1)), self.m_s)
        out_m = self.m_out(self.m_s)

        h1_s = self.h_rnn1(self.identity(torch.cat([h1.unsqueeze(0), out_m[:, :self.osc_hid], self.k1_up], 1)), self.h1_s)
        h2_s = self.h_rnn2(self.identity(torch.cat([h2.unsqueeze(0), out_m[:, self.osc_hid:], self.k2_up], 1)), self.h2_s)

        k1_s = self.k_rnn1(self.identity(torch.cat([k1.unsqueeze(0), self.h1_down, self.f1_up], 1)), self.k1_s)
        k2_s = self.k_rnn2(self.identity(torch.cat([k2.unsqueeze(0), self.h2_down, self.f2_up], 1)), self.k2_s)

        f1_s = self.f_rnn1(self.identity(torch.cat([f1.unsqueeze(0), self.k1_down], 1)), self.f1_s)
        f2_s = self.f_rnn2(self.identity(torch.cat([f2.unsqueeze(0), self.k2_down], 1)), self.f2_s)

        self.h1_s = self.identity(h1_s)
        self.h2_s = self.identity(h2_s)
        self.k1_s = self.identity(k1_s)
        self.k2_s = self.identity(k2_s)
        self.f1_s = self.identity(f1_s)
        self.f2_s = self.identity(f2_s)

        out_h1 = self.h_out1(self.h1_s)
        out_h2 = self.h_out2(self.h2_s)
        h1_down = self.h_out1_down(self.h1_s)
        h1_up = self.h_out1_up(self.h1_s)
        h2_down = self.h_out2_down(self.h2_s)
        h2_up = self.h_out2_up(self.h2_s)

        out_k1 = self.k_out1(self.k1_s)
        out_k2 = self.k_out2(self.k2_s)
        k1_down = self.k_out1_down(self.k1_s)
        k1_up = self.k_out1_up(self.k1_s)
        k2_down = self.k_out2_down(self.k2_s)
        k2_up = self.k_out2_up(self.k2_s)

        out_f1 = self.f_out1(self.f1_s)
        out_f2 = self.f_out2(self.f2_s)
        f1_up = self.f_out1_up(self.f1_s)
        f2_up = self.f_out2_up(self.f2_s)

        self.h1_down = h1_down
        self.h2_down = h2_down
        self.h1_up = h1_up
        self.h2_up = h2_up

        self.k1_down = k1_down
        self.k2_down = k2_down
        self.k1_up = k1_up
        self.k2_up = k2_up

        self.f1_up = f1_up
        self.f2_up = f2_up

        action = torch.cat([out_h1, out_k1, out_f1, out_h2, out_k2, out_f2], 1)

        #print([self.m_s.data, self.h1_s.data, self.h2_s.data, self.k1_s.data, self.k2_s.data, self.f1_s.data, self.f2_s.data])

        return action

    def reset(self):
        # Master
        self.m_s = Variable(torch.zeros(1, self.m_hid)).float()

        # Hip
        self.h1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h1_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h1_down = Variable(torch.zeros(1, self.osc_hid)).float()

        self.h2_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h2_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h2_down = Variable(torch.zeros(1, self.osc_hid)).float()

        # Knee
        self.k1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k1_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k1_down = Variable(torch.zeros(1, self.osc_hid)).float()

        self.k2_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k2_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k2_down = Variable(torch.zeros(1, self.osc_hid)).float()

        # Foot
        self.f1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.f1_up = Variable(torch.zeros(1, self.osc_hid)).float()

        self.f2_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.f2_up = Variable(torch.zeros(1, self.osc_hid)).float()


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class RNet3(nn.Module):
    def __init__(self, m_hid, osc_hid):
        super(RNet3, self).__init__()
        self.m_hid = m_hid
        self.osc_hid = osc_hid

        # Set states
        self.reset()

        # Master
        self.m_rnn = nn.GRUCell(5 + 2 * osc_hid, m_hid)
        self.m_out = nn.Linear(m_hid, 2 * osc_hid)

        # Hip
        self.h_rnn1 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_rnn2 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.h_out1 = nn.Linear(osc_hid, 1)
        self.h_out2 = nn.Linear(osc_hid, 1)
        self.h_out1_down = nn.Linear(osc_hid, osc_hid)
        self.h_out2_down = nn.Linear(osc_hid, osc_hid)
        self.h_out1_up = nn.Linear(osc_hid, osc_hid)
        self.h_out2_up = nn.Linear(osc_hid, osc_hid)

        # Knee
        self.k_rnn1 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_rnn2 = nn.GRUCell(1 + 2 * osc_hid, osc_hid)
        self.k_out1 = nn.Linear(osc_hid, 1)
        self.k_out2 = nn.Linear(osc_hid, 1)
        self.k_out1_down = nn.Linear(osc_hid, osc_hid)
        self.k_out2_down = nn.Linear(osc_hid, osc_hid)
        self.k_out1_up = nn.Linear(osc_hid, osc_hid)
        self.k_out2_up = nn.Linear(osc_hid, osc_hid)

        # Foot
        self.f_rnn1 = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_rnn2 = nn.GRUCell(1 + osc_hid, osc_hid)
        self.f_out1 = nn.Linear(osc_hid, 1)
        self.f_out2 = nn.Linear(osc_hid, 1)
        self.f_out1_up = nn.Linear(osc_hid, osc_hid)
        self.f_out2_up = nn.Linear(osc_hid, osc_hid)

    def normalize(self, tensor):
        tensor_norm = torch.norm(tensor, p=2, dim=1).detach()
        return tensor.div(tensor_norm.expand_as(tensor))

    def identity(self, tensor):
        return tensor

    def forward(self, x):
        #h1, k1, f1, h2, k2, f2, *m_obs = x[0, [2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10]]

        h1 = x[:, 2]
        k1 = x[:, 3]
        f1 = x[:, 4]
        h2 = x[:, 5]
        k2 = x[:, 6]
        f2 = x[:, 7]
        m_obs = x[:, [0, 1, 8, 9, 10]]

        self.m_s = self.m_rnn(self.identity(torch.cat([m_obs , self.h1_up , self.h2_up], 1)), self.m_s)
        out_m = self.m_out(self.m_s)

        h1_s = self.h_rnn1(self.identity(torch.cat([h1.unsqueeze(0), out_m[:, :self.osc_hid], self.k1_up], 1)), self.h1_s)
        h2_s = self.h_rnn2(self.identity(torch.cat([h2.unsqueeze(0), out_m[:, self.osc_hid:], self.k2_up], 1)), self.h2_s)

        k1_s = self.k_rnn1(self.identity(torch.cat([k1.unsqueeze(0), self.h1_down, self.f1_up], 1)), self.k1_s)
        k2_s = self.k_rnn2(self.identity(torch.cat([k2.unsqueeze(0), self.h2_down, self.f2_up], 1)), self.k2_s)

        f1_s = self.f_rnn1(self.identity(torch.cat([f1.unsqueeze(0), self.k1_down], 1)), self.f1_s)
        f2_s = self.f_rnn2(self.identity(torch.cat([f2.unsqueeze(0), self.k2_down], 1)), self.f2_s)

        self.h1_s = self.identity(h1_s)
        self.h2_s = self.identity(h2_s)
        self.k1_s = self.identity(k1_s)
        self.k2_s = self.identity(k2_s)
        self.f1_s = self.identity(f1_s)
        self.f2_s = self.identity(f2_s)

        out_h1 = self.h_out1(self.h1_s)
        out_h2 = self.h_out2(self.h2_s)
        h1_down = self.h_out1_down(self.h1_s)
        h1_up = self.h_out1_up(self.h1_s)
        h2_down = self.h_out2_down(self.h2_s)
        h2_up = self.h_out2_up(self.h2_s)

        out_k1 = self.k_out1(self.k1_s)
        out_k2 = self.k_out2(self.k2_s)
        k1_down = self.k_out1_down(self.k1_s)
        k1_up = self.k_out1_up(self.k1_s)
        k2_down = self.k_out2_down(self.k2_s)
        k2_up = self.k_out2_up(self.k2_s)

        out_f1 = self.f_out1(self.f1_s)
        out_f2 = self.f_out2(self.f2_s)
        f1_up = self.f_out1_up(self.f1_s)
        f2_up = self.f_out2_up(self.f2_s)

        self.h1_down = h1_down
        self.h2_down = h2_down
        self.h1_up = h1_up
        self.h2_up = h2_up

        self.k1_down = k1_down
        self.k2_down = k2_down
        self.k1_up = k1_up
        self.k2_up = k2_up

        self.f1_up = f1_up
        self.f2_up = f2_up

        action = torch.cat([out_h1, out_k1, out_f1, out_h2, out_k2, out_f2], 1)

        #print([self.m_s.data, self.h1_s.data, self.h2_s.data, self.k1_s.data, self.k2_s.data, self.f1_s.data, self.f2_s.data])

        return action

    def reset(self):
        # Master
        self.m_s = Variable(torch.zeros(1, self.m_hid)).float()

        # Hip
        self.h1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h1_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h1_down = Variable(torch.zeros(1, self.osc_hid)).float()

        self.h2_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h2_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.h2_down = Variable(torch.zeros(1, self.osc_hid)).float()

        # Knee
        self.k1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k1_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k1_down = Variable(torch.zeros(1, self.osc_hid)).float()

        self.k2_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k2_up = Variable(torch.zeros(1, self.osc_hid)).float()
        self.k2_down = Variable(torch.zeros(1, self.osc_hid)).float()

        # Foot
        self.f1_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.f1_up = Variable(torch.zeros(1, self.osc_hid)).float()

        self.f2_s = Variable(torch.zeros(1, self.osc_hid)).float()
        self.f2_up = Variable(torch.zeros(1, self.osc_hid)).float()


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.f_up_l1 = nn.Linear(2, 2)
        self.f_up_l2 = nn.Linear(2, 2)
        self.k_up_l1 = nn.Linear(4, 4)
        self.k_up_l2 = nn.Linear(4, 4)
        self.h_up_l1 = nn.Linear(6, 6)
        self.h_up_l2 = nn.Linear(6, 6)

        self.m_down_l1 = nn.Linear(17, 16)
        self.m_down_l2 = nn.Linear(16, 16)
        self.m_down_l2_1 = nn.Linear(16, 6)
        self.m_down_l2_2 = nn.Linear(16, 6)
        self.h_down_l1 = nn.Linear(6, 6)
        self.h_down_l2 = nn.Linear(6, 6)
        self.k_down_l1 = nn.Linear(6, 4)
        self.k_down_l2 = nn.Linear(4, 4)

        self.h_act_l1 = nn.Linear(8, 4)
        self.h_act_l2 = nn.Linear(4, 1)
        self.k_act_l1 = nn.Linear(8, 4)
        self.k_act_l2 = nn.Linear(4, 1)
        self.f_act_l1 = nn.Linear(6, 4)
        self.f_act_l2 = nn.Linear(4, 1)

        torch.nn.init.xavier_uniform_(self.f_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.f_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.k_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.k_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.h_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.h_up_l2.weight)

        torch.nn.init.xavier_uniform_(self.m_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2_1.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2_2.weight)
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

        # ---


        self.xf_up_l1 = nn.Linear(2, 2)
        self.xf_up_l2 = nn.Linear(2, 2)
        self.xk_up_l1 = nn.Linear(4, 4)
        self.xk_up_l2 = nn.Linear(4, 4)
        self.xh_up_l1 = nn.Linear(6, 6)
        self.xh_up_l2 = nn.Linear(6, 6)

        self.xh_down_l1 = nn.Linear(6, 6)
        self.xh_down_l2 = nn.Linear(6, 6)
        self.xk_down_l1 = nn.Linear(6, 4)
        self.xk_down_l2 = nn.Linear(4, 4)

        self.xh_act_l1 = nn.Linear(8, 4)
        self.xh_act_l2 = nn.Linear(4, 1)
        self.xk_act_l1 = nn.Linear(8, 4)
        self.xk_act_l2 = nn.Linear(4, 1)
        self.xf_act_l1 = nn.Linear(6, 4)
        self.xf_act_l2 = nn.Linear(4, 1)


        torch.nn.init.xavier_uniform_(self.xf_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.xf_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.xk_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.xk_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.xh_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.xh_up_l2.weight)

        torch.nn.init.xavier_uniform_(self.xk_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.xk_down_l2.weight)
        torch.nn.init.xavier_uniform_(self.xh_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.xh_down_l2.weight)

        torch.nn.init.xavier_uniform_(self.xh_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.xh_act_l2.weight)
        torch.nn.init.xavier_uniform_(self.xk_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.xk_act_l2.weight)
        torch.nn.init.xavier_uniform_(self.xf_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.xf_act_l2.weight)



    def forward(self, x):

        h1 = x[:, 2].unsqueeze(1)
        k1 = x[:, 3].unsqueeze(1)
        f1 = x[:, 4].unsqueeze(1)
        h2 = x[:, 5].unsqueeze(1)
        k2 = x[:, 6].unsqueeze(1)
        f2 = x[:, 7].unsqueeze(1)
        h1d = x[:, 11].unsqueeze(1)
        k1d = x[:, 12].unsqueeze(1)
        f1d = x[:, 13].unsqueeze(1)
        h2d = x[:, 14].unsqueeze(1)
        k2d = x[:, 15].unsqueeze(1)
        f2d = x[:, 16].unsqueeze(1)
        m_obs = x[:, [0, 1, 8, 9, 10]]

        f_up_1 = F.relu(self.f_up_l2(F.relu(self.f_up_l1(torch.cat([f1, f1d], 1)))))
        k_up_1 = F.relu(self.k_up_l2(F.relu(self.k_up_l1(torch.cat([k1, k1d, f_up_1], 1)))))
        h_up_1 = F.relu(self.h_up_l2(F.relu(self.h_up_l1(torch.cat([h1, h1d, k_up_1], 1)))))

        f_up_2 = F.relu(self.xf_up_l2(F.relu(self.xf_up_l1(torch.cat([f2, f2d], 1)))))
        k_up_2 = F.relu(self.xk_up_l2(F.relu(self.xk_up_l1(torch.cat([k2, k2d, f_up_2], 1)))))
        h_up_2 = F.relu(self.xh_up_l2(F.relu(self.xh_up_l1(torch.cat([h2, h2d, k_up_2], 1)))))

        m_down = F.relu(self.m_down_l2(F.relu(self.m_down_l1(torch.cat([m_obs, h_up_1, h_up_2], 1)))))
        m_down_1 = F.relu(self.m_down_l2_1(m_down))
        m_down_2 = F.relu(self.m_down_l2_2(m_down))

        h_down_1 = F.relu(self.h_down_l2(F.tanh(self.h_down_l1(m_down_1))))
        k_down_1 = F.relu(self.k_down_l2(F.tanh(self.k_down_l1(h_down_1))))

        h_down_2 = F.relu(self.xh_down_l2(F.tanh(self.xh_down_l1(m_down_2))))
        k_down_2 = F.relu(self.xk_down_l2(F.tanh(self.xk_down_l1(h_down_2))))

        h_act_1 = self.h_act_l2(F.relu(self.h_act_l1(torch.cat([h1, h1d, m_down_1], 1))))
        k_act_1 = self.k_act_l2(F.relu(self.k_act_l1(torch.cat([k1, k1d, h_down_1], 1))))
        f_act_1 = self.f_act_l2(F.relu(self.f_act_l1(torch.cat([f1, f1d, k_down_1], 1))))

        h_act_2 = self.xh_act_l2(F.relu(self.xh_act_l1(torch.cat([h2, h2d, m_down_2], 1))))
        k_act_2 = self.xk_act_l2(F.relu(self.xk_act_l1(torch.cat([k2, k2d, h_down_2], 1))))
        f_act_2 = self.xf_act_l2(F.relu(self.xf_act_l1(torch.cat([f2, f2d, k_down_2], 1))))

        act = torch.cat([h_act_1, f_act_1, k_act_1, h_act_2, k_act_2, f_act_2], 1)

        return act

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reset(self):
        pass


class PNet2(nn.Module):

    def __init__(self):
        super(PNet2, self).__init__()

        self.f_up_l1 = nn.Linear(2, 2)
        self.k_up_l1 = nn.Linear(4, 4)
        self.h_up_l1 = nn.Linear(6, 6)

        self.m_down_l1 = nn.Linear(17, 32)
        self.m_down_l2 = nn.Linear(32, 16)
        self.m_down_l2_1 = nn.Linear(16, 6)
        self.m_down_l2_2 = nn.Linear(16, 6)
        self.h_down_l1 = nn.Linear(6, 6)
        self.k_down_l1 = nn.Linear(6, 4)

        self.h_act_l1 = nn.Linear(8, 4)
        self.k_act_l1 = nn.Linear(8, 4)
        self.f_act_l1 = nn.Linear(6, 4)

        torch.nn.init.xavier_uniform_(self.f_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.f_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.k_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.k_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.h_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.h_up_l2.weight)

        torch.nn.init.xavier_uniform_(self.m_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2_1.weight)
        torch.nn.init.xavier_uniform_(self.m_down_l2_2.weight)
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

        # ---


        self.xf_up_l1 = nn.Linear(2, 2)
        self.xf_up_l2 = nn.Linear(2, 2)
        self.xk_up_l1 = nn.Linear(4, 4)
        self.xk_up_l2 = nn.Linear(4, 4)
        self.xh_up_l1 = nn.Linear(6, 6)
        self.xh_up_l2 = nn.Linear(6, 6)

        self.xh_down_l1 = nn.Linear(6, 6)
        self.xh_down_l2 = nn.Linear(6, 6)
        self.xk_down_l1 = nn.Linear(6, 4)
        self.xk_down_l2 = nn.Linear(4, 4)

        self.xh_act_l1 = nn.Linear(8, 4)
        self.xh_act_l2 = nn.Linear(4, 1)
        self.xk_act_l1 = nn.Linear(8, 4)
        self.xk_act_l2 = nn.Linear(4, 1)
        self.xf_act_l1 = nn.Linear(6, 4)
        self.xf_act_l2 = nn.Linear(4, 1)


        torch.nn.init.xavier_uniform_(self.xf_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.xf_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.xk_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.xk_up_l2.weight)
        torch.nn.init.xavier_uniform_(self.xh_up_l1.weight)
        torch.nn.init.xavier_uniform_(self.xh_up_l2.weight)

        torch.nn.init.xavier_uniform_(self.xk_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.xk_down_l2.weight)
        torch.nn.init.xavier_uniform_(self.xh_down_l1.weight)
        torch.nn.init.xavier_uniform_(self.xh_down_l2.weight)

        torch.nn.init.xavier_uniform_(self.xh_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.xh_act_l2.weight)
        torch.nn.init.xavier_uniform_(self.xk_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.xk_act_l2.weight)
        torch.nn.init.xavier_uniform_(self.xf_act_l1.weight)
        torch.nn.init.xavier_uniform_(self.xf_act_l2.weight)



    def forward(self, x):

        h1 = x[:, 2].unsqueeze(1)
        k1 = x[:, 3].unsqueeze(1)
        f1 = x[:, 4].unsqueeze(1)
        h2 = x[:, 5].unsqueeze(1)
        k2 = x[:, 6].unsqueeze(1)
        f2 = x[:, 7].unsqueeze(1)
        h1d = x[:, 11].unsqueeze(1)
        k1d = x[:, 12].unsqueeze(1)
        f1d = x[:, 13].unsqueeze(1)
        h2d = x[:, 14].unsqueeze(1)
        k2d = x[:, 15].unsqueeze(1)
        f2d = x[:, 16].unsqueeze(1)
        m_obs = x[:, [0, 1, 8, 9, 10]]

        f_up_1 = F.relu(self.f_up_l2(F.relu(self.f_up_l1(torch.cat([f1, f1d], 1)))))
        k_up_1 = F.relu(self.k_up_l2(F.relu(self.k_up_l1(torch.cat([k1, k1d, f_up_1], 1)))))
        h_up_1 = F.relu(self.h_up_l2(F.relu(self.h_up_l1(torch.cat([h1, h1d, k_up_1], 1)))))

        f_up_2 = F.relu(self.xf_up_l2(F.relu(self.xf_up_l1(torch.cat([f2, f2d], 1)))))
        k_up_2 = F.relu(self.xk_up_l2(F.relu(self.xk_up_l1(torch.cat([k2, k2d, f_up_2], 1)))))
        h_up_2 = F.relu(self.xh_up_l2(F.relu(self.xh_up_l1(torch.cat([h2, h2d, k_up_2], 1)))))

        m_down = F.relu(self.m_down_l2(F.relu(self.m_down_l1(torch.cat([m_obs, h_up_1, h_up_2], 1)))))
        m_down_1 = F.relu(self.m_down_l2_1(m_down))
        m_down_2 = F.relu(self.m_down_l2_2(m_down))

        h_down_1 = F.relu(self.h_down_l2(F.tanh(self.h_down_l1(m_down_1))))
        k_down_1 = F.relu(self.k_down_l2(F.tanh(self.k_down_l1(h_down_1))))

        h_down_2 = F.relu(self.xh_down_l2(F.tanh(self.xh_down_l1(m_down_2))))
        k_down_2 = F.relu(self.xk_down_l2(F.tanh(self.xk_down_l1(h_down_2))))

        h_act_1 = self.h_act_l2(F.relu(self.h_act_l1(torch.cat([h1, h1d, m_down_1], 1))))
        k_act_1 = self.k_act_l2(F.relu(self.k_act_l1(torch.cat([k1, k1d, h_down_1], 1))))
        f_act_1 = self.f_act_l2(F.relu(self.f_act_l1(torch.cat([f1, f1d, k_down_1], 1))))

        h_act_2 = self.xh_act_l2(F.relu(self.xh_act_l1(torch.cat([h2, h2d, m_down_2], 1))))
        k_act_2 = self.xk_act_l2(F.relu(self.xk_act_l1(torch.cat([k2, k2d, h_down_2], 1))))
        f_act_2 = self.xf_act_l2(F.relu(self.xf_act_l1(torch.cat([f2, f2d, k_down_2], 1))))

        act = torch.cat([h_act_1, f_act_1, k_act_1, h_act_2, k_act_2, f_act_2], 1)

        return act

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def reset(self):
        pass


def evaluate(model, env, iters):
    print("Starting visual evaluation")
    for i in range(iters):
        obs = env.reset()
        done = False
        model.reset()
        total_rew = 0
        with torch.no_grad():
            while not done:
                obs_t = torch.from_numpy(np.array(obs, dtype=np.float32)).unsqueeze(0)
                action = model(obs_t).numpy()
                obs, rew, done, _ = env.step(action + np.random.randn(6) * 0.0)
                total_rew += rew
                env.render()
            print("EV {}/{}, rew: {}".format(i,iters, total_rew))

def train_imitation(model,baseline, trajectories, iters):
    if iters == 0 or iters == None:
        print("Skipping training")
        return

    N = len(trajectories)
    obs_dim = len(trajectories[0][0][0])
    act_dim = len(trajectories[0][0][1])

    print("Starting training. Obs dim: {}, Act dim: {}".format(obs_dim, act_dim))

    lossfun = nn.MSELoss()
    rnn_optim = torch.optim.Adam(model.parameters(), lr=7e-3)
    baseline_optim = torch.optim.Adam(baseline.parameters(), lr=3e-3)


    for i in range(iters):
        # Sample random whole episodes
        rand_episode = trajectories[np.random.randint(0, N)]

        obs_list, act_list = zip(*rand_episode)
        obs_array = torch.from_numpy(np.array(obs_list, dtype=np.float32))
        act_array = torch.from_numpy(np.array(act_list, dtype=np.float32))

        #rnn_pred_list = []

        # Rollout
        model.reset()

        rnn_optim.zero_grad()
        #baseline_optim.zero_grad()
        # for obs in obs_array:
        #     rnn_pred_list.append(model(obs.unsqueeze(0)))

        #rnn_preds = torch.cat(rnn_pred_list, 0)
        rnn_preds = model(obs_array)
        #baseline_preds = baseline(obs_array)

        # MSE & gradients
        rnn_loss = lossfun(rnn_preds, act_array)
        #baseline_loss = lossfun(baseline_preds, act_array)

        rnn_loss.backward()
        #baseline_loss.backward()

        rnn_optim.step()
        #baseline_optim.step()

        if i % 10 == 0:
            print("Iteration: {}/{}, Rnn loss: {}, Baseline loss: {}".format(i, iters, rnn_loss, 0.13))

    torch.save(baseline, 'baseline_imit.pt')
    torch.save(model, 'rnn_imit.pt')

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Baseline model
baseline = Baseline(17, 6)

# RNN
model = PNet()

# Load trajectories
trajectories = pickle.load(open("/home/silverjoda/SW/baselines/data/Walker2d-v2_rollouts_0", 'rb'))

# TODO: VISUALIZE RNN HIDDEN STATE VALUES AND FIX HIDDEN STATE BLOW UP IF NECESSARY
# TODO: PERFORM ROLLOUTS OF TRAINED BASELINES AND TRAINED RNN MODELS TO SEE HOW THE RNN DOES

print("Model params: {}, baseline params: {}".format(count_parameters(model), count_parameters(baseline)))

train_imitation(model, baseline, trajectories, 5000)

env = gym.make("Walker2d-v2")

#print("Evaluating baseline")
#baseline = torch.load('baseline_imit.pt')
#evaluate(baseline, env, 10)
#time.sleep(1)
print("Evaluating Model")
#model = torch.load('rnn_imit.pt')
evaluate(model, env, 10)
