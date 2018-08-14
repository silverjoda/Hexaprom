import numpy as np
import gym
import cma
np.random.seed(0)

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = 8 * [0]

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # Make actions:

        # l0f
        o0 = list(env_obs[[0, 1, 2, 3, 4,   5, 7, 9, 11,   6, 19, 20]]) + prev_torques[0:2]
        t0 = mult * np.tanh(np.matmul(o0, w[0:n1].reshape((14, 1))))

        # l1f
        o2 = list(env_obs[[0, 1, 2, 3, 4,   5, 7, 9, 11,   8, 21, 22]]) + prev_torques[2:4]
        t2 = mult * np.tanh(np.matmul(o2, w[0:n1].reshape((14, 1))))

        # l2f
        o4 = list(env_obs[[0, 1, 2, 3, 4,   5, 7, 9, 11,   10, 23, 24]]) + prev_torques[4:6]
        t4 = mult * np.tanh(np.matmul(o4, w[0:n1].reshape((14, 1))))

        # l3f
        o6 = list(env_obs[[0, 1, 2, 3, 4,   5, 7, 9, 11,   12, 25, 26]]) + prev_torques[6:8]
        t6 = mult * np.tanh(np.matmul(o6, w[0:n1].reshape((14, 1))))

        # l0c
        o1 = list(env_obs[[5, 6]]) + prev_torques[0:2]
        t1 = mult * np.tanh(np.matmul(o1, w[n1:n1+n2].reshape((4, 1))))

        # l1c
        o3 = list(env_obs[[7, 8]]) + prev_torques[2:4]
        t3 = mult * np.tanh(np.matmul(o3, w[n1:n1 + n2].reshape((4, 1))))

        # l2c
        o5 = list(env_obs[[9, 10]]) + prev_torques[4:6]
        t5 = mult * np.tanh(np.matmul(o5, w[n1:n1 + n2].reshape((4, 1))))

        # l3c
        o7 = list(env_obs[[11, 12]]) + prev_torques[6:8]
        t7 = mult * np.tanh(np.matmul(o7, w[n1:n1 + n2].reshape((4, 1))))

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2, t3, t4, t5, t6, t7])

        if animate:
            env.render()

        reward += rew
        prev_torques = [t0, t1, t2, t3, t4, t5, t6, t7]

    return -reward

# Make environment
env = gym.make("Ant-v3")
animate = False

# Generate weights
n1 = (14 * 1)
n2 = (4 * 1)

N_weights = n1 + n2
W_MULT = 0.1
mult = 1
w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("hopper_weights.txt", "a"))
