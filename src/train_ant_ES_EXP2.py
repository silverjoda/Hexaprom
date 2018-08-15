import numpy as np
import gym
import cma
np.random.seed(0)

# EXP:  Try full observation, each joint has all observs
import time

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = 8 * [0]


    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # l0f
        o0 = list(env_obs[[5,6,7,8,9,10,11,12,19,20,21,22,23,24,25,26]]) + list(np.array(prev_torques)[[0,2,4,6]])
        t0 = mult * np.tanh(np.matmul(o0, w[n1:n1 + 10].reshape(n2_s)))

        t0,t1,t2,t3,t4,t5,t6,t7 = np.random.randn(8)
        action = np.array([t0, t1, t2, t3, t4, t5, t6, t7])
        action += np.random.randn(8,1) * 0.3

        # Step environment
        env_obs, rew, done, _ = env.step(action[:,0])

        if animate:
            env.render()

        reward += rew
        prev_torques = [t0, t1, t2, t3, t4, t5, t6, t7]

    return -reward

# Make environment
env = gym.make("Ant-v3")
animate = False

# Generate weights
n1_s = (14, 8)
n2_s = (10, 1)
n3_s = (6, 1)

n1 = n1_s[0] * n1_s[1]
n2 = 4 * n2_s[0] * n2_s[1]
n3 = n3_s[0] * n3_s[1]

N_weights = n1 + n2 + n3
W_MULT = 1
mult = 1
w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=20000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("ant_weights.txt", "a"))
