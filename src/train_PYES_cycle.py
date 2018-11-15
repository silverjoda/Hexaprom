import numpy as np
from numpy import cos
import gym
import cma
from time import sleep
import quaternion
import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ES
import sys


class Baseline(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Baseline, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.n_experts = 4
        self.phase_step = 0.1

        self.experts = nn.ParameterList([nn.Parameter(T.randn(obs_dim, act_dim + 1, requires_grad=True, dtype=T.float32)) for _ in range(self.n_experts)])
        self.phase = 0


    def forward(self, x):

        # TODO: print matrix statistics and decomposition values (eigenvalues etc) during training

        # Blend current weights
        A = T.zeros(self.obs_dim, self.act_dim + 1)
        for i, w in enumerate(self.experts):
            coeff = T.clamp(T.tensor(cos(np.abs(self.phase - i * (np.pi / 4)))), 0, 1).float()
            wi = w.data.float() * coeff
            A += wi

        # Perform forward pass
        x = x.float() @ A

        pred_step = x[:, -1]

        # Step phase
        self.phase = (self.phase + self.phase_step) % (2 * np.pi)

        return x[:, :-1]



def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        pytorch_ES.vector_to_parameters(torch.from_numpy(w), policy.parameters())

        while not done:

            # Remap the observations
            obs = np.concatenate((obs[0:1],obs[8:11],obs[1:8],obs[11:]))

            # Get action from policy
            with torch.no_grad():
                act = policy(torch.from_numpy(np.expand_dims(obs, 0)))[0].numpy()

            # Step environment
            obs, rew, done, _ = env.step(act)

            if animate:
                env.render()

            reward += rew

        return -reward
    return f


def train(params):

    env_name, iters, n_hidden, animate = params

    # Make environment
    env = gym.make(env_name)

    obs_dim, act_dim = env.observation_space.shape[0], env.action_space.shape[0]
    policy = Baseline(obs_dim, act_dim)
    w = pytorch_ES.parameters_to_vector(policy.parameters()).detach().numpy()
    es = cma.CMAEvolutionStrategy(w, 0.5)
    f = f_wrapper(env, policy, animate)

    print("Env: {} Action space: {}, observation space: {}, N_params: {}, comments: ...".format(env_name, env.action_space.shape,
                                                                                  env.observation_space.shape, len(w)))

    try:
        while not es.stop():
            X = es.ask()
            es.tell(X, [f(x) for x in X])
            es.disp()
    except KeyboardInterrupt:
        print("User interrupted process.")

    return es.result.fbest

env_name = "Ant-v3"
train((env_name, 1000, 7, True))
exit()

