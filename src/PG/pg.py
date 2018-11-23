import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import time
import gym

class Policy(nn.Module):
    def __init__(self, env):
        super(Policy, self).__init__()
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.fc1 = nn.Linear(self.obs_dim, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, self.act_dim)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Valuefun(nn.Module):
    def __init__(self, env):
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

    for i in range(params["iters"]):
        s_0 = env.reset()
        done = False

        while not done:
            # Sample action from policy
            action = policy(T.FloatTensor(s_0).unsqueeze(0)).detach()

            # Step action
            s_1, r, done, _ = env.step(action)

            # Record transition
            batch_states.append(s_0)
            batch_actions.append(action)
            batch_rewards.append(T.tensor(r))
            batch_new_states.append(T.FloatTensor(s_1).unsqueeze(0))
            batch_terminals.append(done)

        # Just completed an episode
        batch_ctr += 1

        # If enough data gathered, then perform update
        if batch_ctr == params["batchsize"]:

            # Refit value function
            loss_V = update_V(V, V_optim, params["gamma"], batch_states, batch_rewards, batch_next_states)

            # Calculate episode advantages
            batch_advantages = calc_advantages(V, batch_states, batch_rewards, batch_next_states, batch_terminals)
            batch_advantages = T.FloatTensor(batch_advantages)

            # Update policy
            loss_policy = update_policy(policy, policy_optim, V, batch_states, batch_actions, batch_advantages)

            print("Episode {}/{}, loss_V: {}, loss_policy: {}".format(i, param["iters"], loss_V, loss_policy))

            # Finally reset all batch lists
            batch_ctr = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []


def update_V(V, V_optim, gamma, batch_states, batch_rewards, batch_terminals):
    # Predicted values
    Vs = V(batch_states)

    # Monte carlo estimate of targets
    targets = []
    for i in range(len(batch_states)):
        cumrew = 0
        for j in range(i, len(batch_rewards)):
            cumrew += (gamma ** (j-i)) * batch_rewards[j]
            if batch_terminals[j]:
                break
        targets.append(cumrew)

    # MSE loss
    loss = T.nn.MSELoss(targets - Vs)
    loss.backward()
    V_optim.step()

    return loss.data


def getCumRew(gamma, rewards, terminals, idx):
    gammavec = None
    for i in range(idx, len(rewards)):
        pass



def update_policy(V, batch_states, batch_actions, batch_advantages):

    # Get action log probabilities
    log_probs = policy(batch_action, batch_states)

    # Calculate loss function
    loss = -T.mean(log_probs * batch_advantages)

    # Backward pass on policy
    policy_optimizer.zero_grad()
    loss.backward()

    # Step policy update
    policy_optimizer.step()


def calc_advantages(V, batch_states, batch_rewards, batch_next_states, batch_terminals):
    Vs = V(batch_states)
    Vs_ = V(batch_next_states)
    targets = []
    for s, r, s_, t, vs_ in zip(batch_states, batch_rewards, batch_next_states, batch_terminals, Vs_):
        if t:
            targets.append(r.unsqueeze(0))
        else:
            targets.append(r + gamma * vs_)

    advantages = T.cat(targets) - Vs

    return advantages


if __name__=="__main__":
    env_name = "Hopper-v2"
    env = gym.make(env_name)
    policy = Policy(env)
    V = Valuefun(env)
    params = {"iters" : 1000, "batchsize" : 128, "gamma" : 0.98}
    train(env, policy, V, iters)

    # TODO: debug and test
