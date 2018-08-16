import numpy as np
import gym
import cma
np.random.seed(0)

# EXP: double layer NN, sparser observations (less parameters), heirarchical control

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0]

    while not done:

        # Observations:
        # 0,  1,  2,  3,  4,  5,  6,   7,   8,   9,  10
        # z, th, j1, j2, j3, dx, dz, dth, dj1, dj2, dj3

        # Oscillator 0
        o0 = list(env_obs)
        l1 = np.tanh(np.matmul(o0, w[0:n1].reshape(n1s)))
        l2 = np.tanh(np.matmul(l1, w[n1:n1+n2].reshape(n2s)))
        t0, t1, t2 = np.matmul(l2, w[n1+n2:].reshape(n3s))

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2])

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("Hopper-v2")
animate = False

# Generate weights
n1s = (11, 24)
n2s = (24, 12)
n3s = (12, 3)
n1 = n1s[0] * n1s[1]
n2 = n2s[0] * n2s[1]
n3 = n3s[0] * n3s[1]
N_weights = n1 + n2 + n3
W_MULT = 0.3
ES_STD = 0.5
mult = 3
w = np.random.randn(N_weights) * W_MULT

print("N_weights: {}, mult {}, n_mult {}, ES_ESD {}".format(N_weights, mult, W_MULT, ES_STD))

es = cma.CMAEvolutionStrategy(w, ES_STD)
try:
    es.optimize(f, iterations=20000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("hopper_weights.txt", "a"))
