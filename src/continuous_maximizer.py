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
        self.l1a = nn.Linear(n_hid, obs_dim * 4)
        self.l1b = nn.Linear(n_hid, obs_dim * 2)
        self.out = nn.Linear(obs_dim * 4, obs_dim)
        self.rew = nn.Linear(obs_dim * 2, 1)

        torch.nn.init.xavier_uniform_(self.rnn.weight_hh)
        torch.nn.init.xavier_uniform_(self.rnn.weight_ih)
        torch.nn.init.xavier_uniform_(self.l1a.weight)
        torch.nn.init.xavier_uniform_(self.l1b.weight)
        torch.nn.init.xavier_uniform_(self.out.weight)
        torch.nn.init.xavier_uniform_(self.rew.weight)


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
        self.l1 = nn.Linear(n_hid, obs_dim * 5)
        self.out = nn.Linear(obs_dim * 5, act_dim)

        torch.nn.init.xavier_uniform_(self.rnn.weight_hh)
        torch.nn.init.xavier_uniform_(self.rnn.weight_ih)
        torch.nn.init.xavier_uniform_(self.l1.weight)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def forward(self, x):
        self.h = self.rnn(x, self.h)
        l1 = F.relu(self.l1(self.h))
        return self.out(l1)


    def reset(self, batchsize=1):
        self.h = torch.zeros(batchsize, self.n_hid).float()

def pretrain_model(model, env, iters, lr=1e-3):
    if iters == 0:
        return

    # Train prediction model on random rollouts
    MSE = torch.nn.MSELoss()
    optim_model = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    for i in range(iters):
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
            print("Iter: {}/{}, states_loss: {}, rewards_loss: {}".format(i, iters, loss_states, loss_rewards))

    print("Finished pretraining model on random actions, saving")
    torch.save(model, '{}_model.pt'.format(env.spec.id))


def train_opt(model, policy, env, iters, animate=True, lr_model=1e-3, lr_policy=2e-4, model_rpts=1):
    optim_model = torch.optim.Adam(model.parameters(), lr=lr_model, weight_decay=1e-4)
    optim_policy = torch.optim.Adam(policy.parameters(), lr=lr_policy, weight_decay=1e-4)

    MSE = torch.nn.MSELoss()

    # Training algorithm:
    for i in range(iters):

        ### Policy step ----------------------------------------
        done = False
        s = env.reset()
        policy.reset()
        model.reset()

        reward_predictions = []

        sdiff = torch.zeros(1, env.observation_space.shape[0])
        pred_state = torch.from_numpy(s.astype(np.float32)).unsqueeze(0)

        while not done:

            # Predict action from current state
            pred_a = policy(pred_state)

            # Make prediction
            pred_s, pred_rew = model(torch.cat([torch.from_numpy(s.astype(np.float32)).unsqueeze(0), pred_a], 1))
            reward_predictions.append(pred_rew[0])

            s, rew, done, info = env.step(pred_a.detach().numpy())

            if animate:
                env.render()

            # Difference between predicted state and real
            sdiff = pred_s - torch.from_numpy(s.astype(np.float32))

        # Convert to torch
        rew_pred_tens = torch.stack(reward_predictions)

        # Calculate loss
        policy_score = rew_pred_tens.sum()

        # Backprop
        optim_policy.zero_grad()
        (- policy_score).backward()

        # Update
        optim_policy.step()

        loss_states = 0
        loss_rewards = 0

        ## Model Step ----------------------------------------
        for i in range(model_rpts):

            done = False
            s = env.reset()
            policy.reset()
            model.reset()

            states = []
            rewards = []
            state_predictions = []
            reward_predictions = []

            while not done:

                # Predict action from current state
                with torch.no_grad():
                    pred_a = policy(torch.from_numpy(s.astype(np.float32)).unsqueeze(0)) + torch.randn(1, env.action_space.shape[0]) * 0.2

                # Make prediction
                pred_s, pred_rew = model(torch.cat([torch.from_numpy(s.astype(np.float32)).unsqueeze(0), pred_a], 1))
                state_predictions.append(pred_s[0])
                reward_predictions.append(pred_rew[0])

                s, rew, done, info = env.step(pred_a.detach().numpy())
                rewards.append(rew)
                states.append(s)

                if animate:
                    env.render()

            # Convert to torch
            states_tens = torch.from_numpy(np.asarray(states, dtype=np.float32))
            rewards_tens = torch.from_numpy(np.asarray(rewards, dtype=np.float32)).unsqueeze(1)
            state_pred_tens = torch.stack(state_predictions)
            rew_pred_tens = torch.stack(reward_predictions)

            # Calculate loss
            loss_states = MSE(state_pred_tens, states_tens)
            loss_rewards = MSE(rew_pred_tens, rewards_tens)
            total_model_loss = loss_states + loss_rewards

            # Backprop
            optim_model.zero_grad()
            total_model_loss.backward()

            # Update
            optim_model.step()

        print("Iter: {}/{}, states prediction loss: {}, rew prediction loss: {}, policy score: {}".format(i, iters,
                                                                                                          loss_states,
                                                                                                          loss_rewards,
                                                                                                          policy_score))

def eval(env, policy):

    for i in range(10):
        done = False
        s = env.reset()
        policy.reset()
        rtot = 0
        while not done:
            # Predict action from current state
            pred_a = policy(torch.from_numpy(s.astype(np.float32)).unsqueeze(0))
            s, rew, done, info = env.step(pred_a.detach().numpy())
            env.render()
            rtot += rew

        print("Eval iter {}/{}, rew = {}".format(i, 10, rtot))

def main():

    # Create environment
    env = gym.make("Ant-v3")
    print("Env: {}".format(env.spec.id))
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create prediction model
    model = Model(obs_dim, act_dim, 64)

    # Create policy model
    policy = Policy(obs_dim, act_dim, 64)

    # Pretrain model on random actions
    t1 = time.time()
    pretrain_iters = 5000
    pretrain_model(model, env, pretrain_iters, lr=1e-3)
    if pretrain_iters == 0:
        model = torch.load("{}_model.pt".format(env.spec.id))
        print("Loading pretrained_rnd model")

    print("Pretraining finished, took {} s".format(time.time() - t1))

    # Train optimization
    opt_iters = 300
    train_opt(model, policy, env, opt_iters, animate=True, lr_model=1e-4, lr_policy=5e-3, model_rpts=0)

    # TODO: BATCH TRAING EVERYTHING. SINGLE EXAMPLE UPDATES TOO NOISY
    # TODO: TRY WITH AND WITHOUT THE DIFF
    # TODO: TRY STOCHASTIC ACTIONS

    print("Finished training, saving")
    torch.save(policy, '{}_policy.pt'.format(env.spec.id))
    torch.save(model, '{}_model.pt'.format(env.spec.id))

    if opt_iters == 0:
        policy = torch.load("{}_policy.pt".format(env.spec.id))
        print("Loading pretrained_policy")

    eval(env, policy)

if __name__=='__main__':
    main()