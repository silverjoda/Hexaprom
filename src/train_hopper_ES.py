import numpy as np
import gym
import cma
from weight_distributor import Wdist
from time import sleep
import quaternion

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()

    while not done:

        # Observations
        l1 = np.matmul(np.asarray(env_obs), wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w)
        #l2 = np.tanh(np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w))
        #l3 = np.matmul(l2, wdist.get_w('w_l3', w)) + wdist.get_w('b_l3', w)

        # Step environment
        env_obs, rew, done, _ = env.step(l1)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
import roboschool
env = gym.make("Humanoid-v2")
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

N = env.action_space.shape[0]

# Generate weights
wdist = Wdist()

afun = np.tanh
actfun = lambda x:x

print("afun: {}".format(afun))

wdist.addW((env.observation_space.shape[0], env.action_space.shape[0]), 'w_l1')
wdist.addW((env.action_space.shape[0],), 'b_l1')

# wdist.addW((6, 6), 'w_l2')
# wdist.addW((6,), 'b_l2')
#
# wdist.addW((6, 3), 'w_l3')
# wdist.addW((3,), 'b_l3')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
w = np.random.randn(N_weights)

es = cma.CMAEvolutionStrategy(w, 0.5)

print("Comments: ...")

try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, es.result.fbest, sep=',', file=open("hopper_weights.txt", "a"))


