import numpy as np
import gym
import cma
from weight_distributor import Wdist
from time import sleep
import quaternion

def relu(x):
    return np.maximum(x,0)


# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()

    while not done:
        #env_obs = np.concatenate((env_obs[5:17],env_obs[23:]))

        # Observations
        l1 = np.tanh(np.matmul(np.asarray(env_obs), wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w))
        l2 = np.tanh(np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w))
        l3 = np.matmul(l2, wdist.get_w('w_l3', w)) + wdist.get_w('b_l3', w)

        # Step environment
        env_obs, rew, done, _ = env.step(l3)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("Hexapod-v0")
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

N = env.action_space.shape[0]

# Generate weights
wdist = Wdist()

afun = np.tanh
actfun = lambda x:x

print("afun: {}".format(afun))

wdist.addW((47, 8), 'w_l1')
wdist.addW((8,), 'b_l1')

wdist.addW((8, 8), 'w_l2')
wdist.addW((8,), 'b_l2')

wdist.addW((8, 18), 'w_l3')
wdist.addW((18,), 'b_l3')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
w = np.random.randn(N_weights)


print("Comments: relu")
es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=7000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, es.result.fbest, sep=',', file=open("hopper_weights.txt", "a"))


