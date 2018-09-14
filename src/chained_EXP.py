import numpy as np
import gym
import cma
np.random.seed(0)

from weight_distributor import Wdist
np.random.seed(0)
from time import sleep
import quaternion

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()

    prev_torques = [0] * 7

    while not done:

        # Observations
        #  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17
        # th, j1, j2, j3, j4, j5, j6, j7, dx, dy, dth, dj1, dj2, dj3, dj4, dj5, dj6, dj7

        torques = []

        for i in range(N):
            if i == 0:
                nt = [0] + prev_torques[1:2]
            elif i == N - 1:
                nt = prev_torques[N-1:N] + [0]
            else:
                nt = prev_torques[i-1:i] + prev_torques[i+1:i+2]
            obs = list(env_obs[i:i+1]) + nt
            l1 = np.tanh(np.matmul(obs, wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w))
            t0 = np.tanh(np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w))
            torques.append(t0)

        # -------------

        # Step environment
        env_obs, rew, done, _ = env.step(torques)

        if animate:
            env.render()

        reward += rew
        prev_torques = torques

    return -reward

# Make environment
env = gym.make("SwimmerLong-v0")
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

N = env.action_space.shape[0]

# Generate weights
wdist = Wdist()

afun = np.tanh
actfun = lambda x:x

print("afun: {}".format(afun))

wdist.addW((3, 3), 'w_l1')
wdist.addW((3,), 'b_l1')

wdist.addW((3, 1), 'w_l2')
wdist.addW((1,), 'b_l2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
w = np.random.randn(N_weights) * 0.5

es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, es.result.fbest, sep=',', file=open("long_swimmer_weights.txt", "a"))


