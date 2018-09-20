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

        self.rnn = nn.GRUCell(obs_dim + act_dim, n_hid)
        self.l1a = nn.Linear(n_hid, 32)
        self.l1b = nn.Linear(n_hid, 16)
        self.out = nn.Linear(32, obs_dim)
        self.rew = nn.Linear(16, 1)


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

        self.rnn = nn.GRUCell(obs_dim, n_hid)
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
    model = Model(obs_dim, act_dim, 16)

    # Create policy model
    policy = Policy(obs_dim, act_dim, 16)

    # Train prediction model on random rollouts
    MSE = torch.nn.MSELoss()
    optim_model = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model_eps = 0
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

            # Make prediction
            sa = np.concatenate([np.expand_dims(s, 0), np.expand_dims(a, 0)], axis=1).astype(np.float32)
            pred, rew = model(torch.from_numpy(sa))
            state_predictions.append(pred[0])
            reward_predictions.append(rew[0])

            s, rew, done, info = env.step(a)
            rewards.append(rew)
            states.append(s)

        # Convert to torch tensors
        states_tens = torch.from_numpy(np.asarray(states, dtype=np.float32))
        rewards_tens = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1)
        state_pred_tens = torch.stack(state_predictions)
        rew_pred_tens = torch.stack(reward_predictions)

        # Calculate loss
        loss_states = MSE(state_pred_tens, states_tens)
        loss_rewards = MSE(rew_pred_tens, rewards_tens)
        total_loss = loss_rewards + loss_states

        # Backprop
        optim_model.zero_grad()
        total_loss.backward()

        # Update
        optim_model.step()
        if i % 10 == 0:
            print("Iter: {}/{}, states_loss: {}, rewards_loss: {}".format(i, model_eps, loss_states, loss_rewards))

    print("Finished training model")

    # Training algorithm:
    optim_policy = torch.optim.Adam(policy.parameters(), lr=1e-3, weight_decay=1e-4)
    states = []
    rewards = []
    state_predictions = []
    reward_predictions = []
    trn_eps = 10
    animate = False
    for i in range(trn_eps):
        done = False
        s = env.reset()
        model.reset()
        policy.reset()

        sdiff = torch.zeros(1, obs_dim)
        pred_state = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)

        while not done:

            # Predict action from current state
            pred_a = policy(pred_state - sdiff)

            # Make prediction
            pred_s, pred_rew = model(torch.cat([torch.from_numpy(s.astype(np.float32)).unsqueeze(0), pred_a], 1))
            state_predictions.append(pred_s[0])
            reward_predictions.append(pred_rew[0])

            s, rew, done, info = env.step(pred_a.detach().numpy())
            rewards.append(rew)
            states.append(s)

            if animate:
                env.render()

            # Difference between predicted state and real
            sdiff = pred_s - torch.from_numpy(s.astype(np.float32))

        # Convert to torch
        states_tens = torch.from_numpy(np.asarray(states, dtype=np.float32))
        rewards_tens = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1)
        state_pred_tens = torch.stack(state_predictions)
        rew_pred_tens = torch.stack(reward_predictions)

        # Calculate loss
        loss_states = MSE(state_pred_tens, states_tens)
        loss_rewards = MSE(rew_pred_tens, rewards_tens)
        total_model_loss = loss_rewards + loss_states
        policy_loss = -rew_pred_tens.sum()

        # Backprop
        optim_model.zero_grad()
        optim_policy.zero_grad()
        total_model_loss.backward()
        policy_loss.backward()

        # Update
        optim_model.step()
        optim_policy.step()

        print("Iter: {}/{}, model_loss: {}, policy_loss: {}".format(i, trn_eps, model_loss, policy_loss))

if __name__=='__main__':
    main()