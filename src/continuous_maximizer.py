import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
        self.l1a = nn.Linear(n_hid, 32)
        self.l1b = nn.Linear(n_hid, 16)
        self.out = nn.Linear(32, obs_dim)
        self.rew = nn.Linear(16, obs_dim)


    def forward(self, x):
        self.h = self.rnn(x, self.h)
        l1a = F.relu(self.l1a(self.h))
        l1b = F.relu(self.l1b(self.h))
        return self.out(l1a), self.rew(l1b)


    def reset(self, batchsize=1):
        self.h = torch.zeros(batchsize, self.n_hid).float()


class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, n_hid):
        super(Policy, self).__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_hid = n_hid

        # Set states
        self.reset()

        self.rnn_out = nn.GRUCell(obs_dim, n_hid)
        self.l1 = nn.Linear(n_hid, 32)
        self.out = nn.Linear(32, act_dim)


    def forward(self, x):
        self.h = self.rnn(x, self.h)
        l1 = F.relu(self.l1(self.h))
        return self.out(l1)


    def reset(self, batchsize=1):
        self.h = torch.zeros(batchsize, self.n_hid).float()


def main():

    # Create environment
    env = gym.make("Hopper-v2")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create prediction model
    model = Model(obs_dim, act_dim, obs_dim)

    # Create policy model
    policy = Policy(obs_dim, act_dim, obs_dim)

    # Train prediction model on random rollouts
    MSE = torch.nn.MSELoss()
    optim_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model_eps = 10
    for i in range(model_eps):
        s = env.reset()
        model.reset()
        done = False

        states = []
        rewards = []
        state_predictions = []
        reward_predictions = []
        while not done:
            a = env.action_space.sample()
            states.append(s)

            # Make prediction
            pred, rew = model(torch.cat([np.expand_dims(np.asarray(s),0), np.expand_dims(np.asarray(a),0)], 1))
            state_predictions.append(pred)
            reward_predictions.append(rew)

            s, rew, done, info = env.step(a)
            rewards.append(rew)

        # Convert to torch tensors
        newstates_tens = torch.from_numpy(np.asarray(states[1:], dtype=np.float32))
        rewards_tens = torch.from_numpy(np.asarray(rewards, dtype=np.float32))
        state_pred_tens = torch.stack(state_predictions)
        rew_pred_tens = torch.stack(reward_predictions)

        # Calculate loss
        loss_states = MSE(state_pred_tens, newstates_tens)
        loss_rewards = MSE(rew_pred_tens, rewards_tens)
        total_loss = loss_rewards + loss_states

        # Backprop
        optim_model.zero_grad()
        total_loss.backward()

        # Update
        optim_model.step()
        print("Iter: {}/{}, states_loss: {}, rewards_loss: {}".format(i, model_eps, loss_states, loss_rewards))

    print("Finished training model")

    # Training algorithm:
    optim_policy = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    states = []
    state_predictions = []
    rewards = []
    trn_eps = 10
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

        print("Iter: {}/{}, model_loss: {}, policy_loss: {}".format(i, trn_eps, model_loss, policy_loss))

if __name__=='__main__':
    main()