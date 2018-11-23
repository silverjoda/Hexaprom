import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import time
import gym

class Policy:
    def __init__(self):
        pass

class Valuefun:
    def __init__(self):
        pass

def train(env, policy, V, params):

    policy_optimizer = T.optim.Adam(lr=params["policy_lr"])
    V_optimizer = T.optim.Adam(lr=params["V_lr"])

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
            action = policy.sample(T.FloatTensor(s_0).unsqueeze(0)).detach()

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
            loss_V = update_V(batch_states, batch_rewards, batch_next_states)

            # Update policy
            loss_policy = update_policy(V, batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals)

            print("Episode {}/{}, loss_V: {}, loss_policy: {}".format(i, param["iters"], loss_V, loss_policy))

            # Finally reset all batch lists
            batch_ctr = 0

            batch_states = []
            batch_actions = []
            batch_rewards = []
            batch_new_states = []
            batch_terminals = []

def update_V(batch_states, batch_rewards, batch_next_states):
    pass


def update_policy(V, batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals):
    # Calculate episode advantages
    batch_advantages = calc_advantages(V, batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals)
    batch_advantages = T.FloatTensor(batch_advantages)

    # Get action log probabilities
    log_probs = policy(batch_action, batch_states)

    # Calculate loss function
    loss = -T.mean(log_probs * batch_advantages)

    # Backward pass on policy
    policy_optimizer.zero_grad()
    loss.backward()

    # Step policy update
    policy_optimizer.step()

def calc_advantages(batch_states, batch_actions, batch_rewards, batch_next_states, batch_terminals):
    pass



if __name__=="__main__":
    env_name = "Hopper-v2"
    env = gym.make(env_name)
    policy = Policy(env)
    V = Valuefun(env)
    params = {"iters" : 1000, "batchsize" : 128, "gamma" : 0.98}
    train(env, policy, V, iters)
