import numpy as np
import gym
import cma
from pid import PDreg
from weight_distributor import Wdist
np.random.seed(0)
from time import sleep

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()

    while not done:

        # fl
        obs = env_obs
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        act = softmax(np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w))

        # Step environment
        env_obs, rew, done, _ = env.step(np.argmax(act))

        if animate:
            env.render()

        reward += rew


    return -reward

# Make environment
env = gym.make("Acrobot-v1")
animate = False

# Generate weights
wdist = Wdist()
wdist.addW((6, 12), 'w1')
wdist.addW((12,), 'b1')
wdist.addW((12, 3), 'w2')
wdist.addW((3,), 'b2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("invpend_weights.txt", "a"))
