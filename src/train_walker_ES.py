import numpy as np
import gym
import cma
from pid import PDreg
np.random.seed(0)

# EXP:  CLone experiment and add control heirarchy (first policy controls femurs, femurs control coxas)

def f(w):

    w0 = w[0:n1_s[0]].reshape(n1_s)
    w1 = w[n1_s[0]:2*n1_s[0]].reshape(n1_s)

    w2 = w[n1:n1 + n2_s[0]].reshape(n2_s)
    w3 = w[n1 + n2_s[0]:n1 + n2].reshape(n2_s)

    w4 = w[n1 + n2:n1 + n2 + n3_s[0]].reshape(n3_s)
    w5 = w[n1 + n2 + n3_s[0]:n1 + n2 + n3].reshape(n3_s)

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = 6 * [0]

    pdreg = PDreg(1.5, 0.005)

    # Observations
    # 0,  1,   2,   3,   4,   5,   6,   7,  8,  9,  10,   11,   12,   13,   14,   15,   16
    # z, th, lj0, lj1, lj2, rj0, rj1, rj2, dx, dz, dth, dlj0, dlj1, dlj2, drj0, drj1, drj2

    while not done:

        # lj0
        o0 = list(env_obs[[0, 1, 2, 5, 8, 9, 11]]) + list(np.array(prev_torques)[[0, 3]])
        t0 = mult * np.tanh(np.matmul(o0, w0))

        # lj1
        o1 = list(env_obs[[2, 3, 4, 12]]) + list(np.array(prev_torques)[[1, 4]]) + [t0]
        t1 = mult * np.tanh(np.matmul(o1, w2))

        # lj2
        o2 = list(env_obs[[2, 3, 4, 13]]) + list(np.array(prev_torques)[[2]]) + [t0, t1]
        t2 = mult * np.tanh(np.matmul(o2, w4))

        # rj0
        o3 = list(env_obs[[0, 1, 5, 2, 9, 8, 14]]) + list(np.array(prev_torques)[[3, 0]])
        t3 = mult * np.tanh(np.matmul(o3, w1))

        # rj1
        o4 = list(env_obs[[5, 6, 7, 15]]) + list(np.array(prev_torques)[[4, 1]]) + [t3]
        t4 = mult * np.tanh(np.matmul(o4, w3))

        # rj2
        o5 = list(env_obs[[5, 6, 7, 16]]) + list(np.array(prev_torques)[[5]]) + [t3, t4]
        t5 = mult * np.tanh(np.matmul(o5, w5))

        action = [t0[0], t1[0], t2[0], t3[0], t4[0], t5[0]]

        a0, a1, a2, a3, a4, a5 = pdreg.update(env_obs[[2, 3, 4, 5, 6, 7]], action)

        # Step environment
        env_obs, rew, done, _ = env.step([a0, a1, a2, a3, a4, a5])

        if animate:
            env.render()

        reward += rew
        prev_torques = action

    return -reward

# Make environment
env = gym.make("Walker2d-v2")
animate = False

# Generate weights
n1_s = (9, 1)
n2_s = (7, 1)
n3_s = (7, 1)

n1 = 2 * n1_s[0] * n1_s[1]
n2 = 2 * n2_s[0] * n2_s[1]
n3 = 2 * n3_s[0] * n3_s[1]

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


print(es.result.xbest, file=open("walker_weights.txt", "a"))
