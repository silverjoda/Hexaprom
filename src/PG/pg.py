import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import time
import gym
import utils
from copy import deepcopy

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.obs_dim, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, self.act_dim)

        self.log_std = nn.Parameter(T.zeros(1, self.act_dim))


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


    def sample_action(self, s):
        return T.normal(self.forward(s), T.exp(self.log_std))


    def log_probs(self, batch_states, batch_actions):
        # Get action means from policy
        action_means = self.forward(batch_states)

        # Calculate probabilities
        log_std_batch = self.log_std.expand_as(action_means)
        std = T.exp(log_std_batch)
        var = std.pow(2)
        log_density = - T.pow(batch_actions - action_means, 2) / (2 * var) - 0.5 * np.log(2 * np.pi) - log_std_batch

        return log_density.sum(1, keepdim=True)


class Valuefun(nn.Module):
    def __init__(self, env):
        super(Valuefun, self).__init__()

        self.obs_dim = env.observation_space.shape[0]

        self.fc1 = nn.Linear(self.obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(env, policy, V, params):

    policy_optim = T.optim.Adam(policy.parameters(), lr=params["policy_lr"])
    V_optim = T.optim.Adam(V.parameters(), lr=params["V_lr"])

    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_new_states = []
    batch_terminals = []

    batch_ctr = 0
    batch_rew = 0

    for i in range(params["iters"]):
        s_0 = env.reset()
        done = False

        while not done:
            # Sample action from policy
            action = policy.sample_action(utils.to_tensor(s_0, True)).detach()

            # Step action
            s_1, r, done, _ = env.step(action.squeeze(0).numpy())

            batch_rew += r

            if params["animate"]:
                env.render()

            # Record transition
            batch_states.append(utils.to_tensor(s_0, True))
            batch_actions.append(action)
            batch_rewards.append(utils.to_tensor(np.asarray(r, dtype=np.float32), True))
            batch_new_states.append(utils.to_tensor(s_1, True))
            batch_terminals.append(done)

            s_0 = s_1

        # Just completed an episode
        batch_ctr += 1

        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:

            batch_states = T.cat(batch_states)
            batch_actions = T.cat(batch_actions)
            batch_rewards = T.cat(batch_rewards)
            batch_new_states = T.cat(batch_new_states)

            # Refit value function
            #loss_V = update_V(V, V_optim, params["gamma"], batch_states, batch_rewards, batch_terminals)
            loss_V = None

            # Calculate episode advantages
            #batch_advantages = calc_advantages(V, params["gamma"], batch_states, batch_rewards, batch_new_states, batch_terminals)
            batch_advantages = calc_advantages_MC(params["gamma"], batch_rewards, batch_terminals)

            #update_ppo(policy, batch_states, batch_actions, batch_advantages)

            # Update policy
            loss_policy = update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages)
            #loss_policy = None

            print("Episode {}/{}, loss_V: {}, loss_policy: {}, mean ep_rew: {}, std: {}".format(i, params["iters"], loss_V, loss_policy, batch_rew / params["batchsize"], T.exp(policy.log_std).detach().numpy()))

            # Finally reset all batch lists
            batch_ctr = 0
            batch_rew = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []


def update_ppo(policy, batch_states, batch_actions, batch_advantages, policy_optim):
    log_probs_old = policy.log_probs(batch_states, batch_actions).detach()

    # Do ppo_update
    for k in range(8):
        c_eps = 0.2
        log_probs_new = policy.log_probs(batch_states, batch_actions)
        r = log_probs_new / log_probs_old
        loss = -T.mean(T.min(r * batch_advantages, r.clamp(1 - c_eps, 1 + c_eps) * batch_advantages))
        policy_optim.zero_grad()
        loss.backward()
        policy_optim.step()


def update_V(V, V_optim, gamma, batch_states, batch_rewards, batch_terminals):
    assert len(batch_states) == len(batch_rewards) == len(batch_terminals)
    N = len(batch_states)

    # Predicted values
    Vs = V(batch_states)

    # Monte carlo estimate of targets
    targets = []
    for i in range(N):
        cumrew = T.tensor(0.)
        for j in range(i, N):
            cumrew += (gamma ** (j-i)) * batch_rewards[j]
            if batch_terminals[j]:
                break
        targets.append(cumrew.view(1, 1))

    targets = T.cat(targets)

    # MSE loss#
    V_optim.zero_grad()

    loss = (targets - Vs).pow(2).mean()
    loss.backward()
    V_optim.step()

    return loss.data


def update_policy(policy, policy_optim, batch_states, batch_actions, batch_advantages):

    # Get action log probabilities
    log_probs = policy.log_probs(batch_states, batch_actions)

    # Calculate loss function
    loss = -T.mean(log_probs * batch_advantages)

    # Backward pass on policy
    policy_optim.zero_grad()
    loss.backward()

    # Step policy update
    policy_optim.step()

    return loss.data


def calc_advantages(V, gamma, batch_states, batch_rewards, batch_next_states, batch_terminals):
    Vs = V(batch_states)
    Vs_ = V(batch_next_states)
    targets = []
    for s, r, s_, t, vs_ in zip(batch_states, batch_rewards, batch_next_states, batch_terminals, Vs_):
        if t:
            targets.append(r.unsqueeze(0))
        else:
            targets.append(r + gamma * vs_)

    return T.cat(targets) - Vs


def calc_advantages_MC(gamma, batch_rewards, batch_terminals):
    N = len(batch_rewards)

    # Monte carlo estimate of targets
    targets = []
    for i in range(N):
        cumrew = T.tensor(0.)
        for j in range(i, N):
            cumrew += (gamma ** (j - i)) * batch_rewards[j]
            if batch_terminals[j]:
                break
        targets.append(cumrew.view(1, 1))
    targets = T.cat(targets)

    return targets


if __name__=="__main__":
    env_name = "Hopper-v2"
    env = gym.make(env_name)
    policy = Policy(env)
    V = Valuefun(env)
    params = {"iters" : 100000, "batchsize" : 64, "gamma" : 0.99, "policy_lr" : 0.007, "V_lr" : 0.007, "animate" : True}
    train(env, policy, V, params)

    # TODO: debug and test
