import numpy as np
import gym
import cma
np.random.seed(0)

# EXP:  CLone experiment and add control heirarchy (first policy controls femurs, femurs control coxas)

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = 8 * [0]

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # cpol
        ocpol = list(env_obs[[0, 1, 2, 3, 5, 7, 9, 11, 13, 14, 15, 16, 17, 18]])
        pcpol = np.tanh(np.matmul(ocpol, w[:n1].reshape(n1_s)))

        # l0f
        o0 = list(pcpol[0:2]) + list(env_obs[[5, 6, 19, 20]]) + list(np.array(prev_torques)[[0,2,4,6]])
        t0 = mult * np.tanh(np.matmul(o0, w[n1:n1 + 10].reshape(n2_s)))

        # l1f
        o2 = list(pcpol[2:4]) + list(env_obs[[7, 8, 21, 22]]) + list(np.array(prev_torques)[[0,2,4,6]])
        t2 = mult * np.tanh(np.matmul(o2, w[n1 + 10:n1 + 20].reshape(n2_s)))

        # l2f
        o4 = list(pcpol[4:6]) + list(env_obs[[9, 10, 23, 24]]) + list(np.array(prev_torques)[[0,2,4,6]])
        t4 = mult * np.tanh(np.matmul(o4, w[n1 + 20:n1 + 30].reshape(n2_s)))

        # l3f
        o6 = list(pcpol[6:8]) + list(env_obs[[11, 12, 25, 26]]) + list(np.array(prev_torques)[[0,2,4,6]])
        t6 = mult * np.tanh(np.matmul(o6, w[n1 + 30:n1 + 40].reshape(n2_s)))

        # l0c
        o1 = list(env_obs[[5, 6, 19, 20]]) + [t0] + prev_torques[1:2]
        t1 = mult * np.tanh(np.matmul(o1, w[n1 + n2:].reshape(n3_s)))

        # l1c
        o3 = list(env_obs[[7, 8, 21, 22]]) + [t2] + prev_torques[3:4]
        t3 = mult * np.tanh(np.matmul(o3, w[n1 + n2:].reshape(n3_s)))

        # l2c
        o5 = list(env_obs[[9, 10, 23, 24]]) + [t4] + prev_torques[5:6]
        t5 = mult * np.tanh(np.matmul(o5, w[n1 + n2:].reshape(n3_s)))

        # l3c
        o7 = list(env_obs[[11, 12, 25, 26]]) + [t6] + prev_torques[7:8]
        t7 = mult * np.tanh(np.matmul(o7, w[n1 + n2:].reshape(n3_s)))

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
