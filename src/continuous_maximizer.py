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
        self.l1a = nn.Linear(n_hid, 16)
        self.l1b = nn.Linear(n_hid, 16)
        self.out = nn.Linear(16, obs_dim)
        self.rew = nn.Linear(16, obs_dim)


    def forward(self, x):
        self.h = self.rnn(x, self.h)
        l1a = self.l1a(self.h)
        l1b = self.l1b(self.h)
        return self.out(l1a), self.rew(l1b)


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
    loss_fun_model = torch.nn.MSELoss()
    optim_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model_eps = 1000
    for i in range(model_eps):
        s = env.reset()
        model.reset()
        done = False

        states = []
        predictions = []
        while not done:
            a = env.action_space.sample()
            states.append(s)

            # Make prediction
            pred = model(torch.cat([np.expand_dims(np.asarray(s),0), np.expand_dims(np.asarray(a),0)], 1))
            predictions.append(pred)

            s, rew, done, info = env.step(a)

        newstates = np.asarray(states[1:], dtype=np.float32)

        # Convert to torch tensors
        newstates_tens = torch.from_numpy(newstates)
        pred_tens = torch.stack(predictions)

        # Calculate loss
        loss = loss_fun_model(pred_tens, newstates_tens)

        # Backprop
        optim_model.zero_grad()
        loss.backward()

        # Update
        optim_model.step()


    # Training algorithm:
    optim_policy = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    states = []
    state_predictions = []
    rewards = []
    trn_eps = 1000
    for i in range(trn_eps):

        done = False
        s = env.reset()
        model.reset()
        policy.reset()

        sdiff = torch.zeros(1, obs_dim)
        pred_state = torch.from_numpy(s).unsqueeze(0)

        while not done:
            states.append(s)

            # Predict action from current state
            pred_a = policy(pred_state - sdiff)

            # Predict next state from current state and predicted action
            pred_s, pred_rew = model(torch.cat([torch.from_numpy(s).unsqueeze(0), pred_a]))
            state_predictions.append(pred_s)

            # Perform simulation to get real state
            s = env.step(pred_a.numpy())

            # Difference between predicted state and real
            sdiff = pred_s - torch.from_numpy(s)

        newstates = np.asarray(states[1:], dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)

        # Convert to torch tensors
        newstates_tens = torch.from_numpy(newstates)
        rewards_tens = torch.from_numpy(rewards)
        pred_tens = torch.stack(state_predictions)

        # Calculate loss
        model_loss = loss_fun_model(pred_tens, newstates_tens)
        policy_loss = -rewards_tens.sum() # Just x coordinate

        # Backprop
        optim_model.zero_grad()
        optim_policy.zero_grad()
        model_loss.backward()
        policy_loss.backward()

        # Update
        optim_model.step()
        optim_policy.step()

if __name__=='__main__':
    main()