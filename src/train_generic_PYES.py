import numpy as np
import gym
import cma
from time import sleep
import quaternion
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_ES


class Baseline(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Baseline, self).__init__()

        self.fc1 = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        return x


def f_wrapper(env, policy, animate):
    def f(w):
        reward = 0
        done = False
        obs = env.reset()

        # Inject current parameters into policy ## CONSIDER MAKING HARD COPY OF POLICY HERE NOT TO INTERFERE WITH INITIAL POLICY ##
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

env_name = "SwimmerLong-v0"
train((env_name, 1000, 7, True))
exit()

