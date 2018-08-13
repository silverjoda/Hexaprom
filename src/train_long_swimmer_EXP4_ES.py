import numpy as np
import gym
import cma
np.random.seed(0)

#EXP: no oscillator, just torque prediction + double layer NN

# the function we want to optimize
def f(w):

    reward = 0
    step = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0,0,0,0,0]
    prev_states = [0,0,0,0,0,0,0]

    while not done:

        # Observations
        #  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,  11,  12,  13,  14,  15,  16,  17
        # th, j1, j2, j3, j4, j5, j6, j7, dx, dy, dth, dj1, dj2, dj3, dj4, dj5, dj6, dj7

        # Make actions:

        mult = 1.5

        # (neck)
        o0 = list(env_obs[[0,1,8,9,10,11]]) + prev_states[0:2] + prev_torques[0:1]
        t0 = np.tanh(np.matmul(o0, w[0: n1_a].reshape((9, 3))))
        t0, s0 = np.tanh(np.matmul(t0, w[n1_a: n1_a + n1_b].reshape((3, 2))))

        # -------------

        # Oscillator 1
        o1 = list(env_obs[[1, 2, 3, 11, 12, 13]]) + prev_states[0:3] + prev_torques[1:2]
        t1 = np.tanh(np.matmul(o1, w[n1:n1 + n2_a].reshape((10, 3))))
        t1, s1 = mult * np.tanh(np.matmul(t1, w[n1 + n2_a:N_weights].reshape((3, 2))))

        # Oscillator 2
        o2 = list(env_obs[[2, 3, 4, 12, 13, 14]]) + prev_states[1:4] + prev_torques[2:3]
        t2 = np.tanh(np.matmul(o2, w[n1:n1 + n2_a].reshape((10, 3))))
        t2, s2 = mult * np.tanh(np.matmul(t2, w[n1 + n2_a:N_weights].reshape((3, 2))))

        # Oscillator 3
        o3 = list(env_obs[[3, 4, 5, 13, 14, 15]]) + prev_states[2:5] + prev_torques[3:4]
        t3 = np.tanh(np.matmul(o3, w[n1:n1 + n2_a].reshape((10, 3))))
        t3, s3 = mult * np.tanh(np.matmul(t3, w[n1 + n2_a:N_weights].reshape((3, 2))))

        # Oscillator 4
        o4 = list(env_obs[[4, 5, 6, 13, 14, 15]]) + prev_states[2:5] + prev_torques[4:5]
        t4 = np.tanh(np.matmul(o4, w[n1:n1 + n2_a].reshape((10, 3))))
        t4, s4 = mult * np.tanh(np.matmul(t4, w[n1 + n2_a:N_weights].reshape((3, 2))))

        # Oscillator 5
        o5 = list(env_obs[[5, 6, 7, 14, 15, 16]]) + prev_states[3:6] + prev_torques[5:6]
        t5 = np.tanh(np.matmul(o5, w[n1:n1 + n2_a].reshape((10, 3))))
        t5, s5 = mult * np.tanh(np.matmul(t5, w[n1 + n2_a:N_weights].reshape((3, 2))))

        # Oscillator 6
        o6 = list(env_obs[[6, 7]]) + [0] + list(env_obs[[15, 16]]) + [0] + prev_states[4:6] + [0] + prev_torques[6:7]
        t6 = np.tanh(np.matmul(o6, w[n1:n1 + n2_a].reshape((10, 3))))
        t6, s6 = mult * np.tanh(np.matmul(t6, w[n1 + n2_a:N_weights].reshape((3, 2))))

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2, t3, t4, t5, t6])

        if animate:
            env.render()

        reward += rew
        step += 0.03

        prev_torques = [t0, t1, t2, t3, t4, t5, t6]

    return -reward

# Make environment
env = gym.make("SwimmerLong-v0")
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

# Generate weights
n1_a = (9 * 3)
n1_b = (3 * 2)
n1 = n1_a + n1_b
n2_a = (10 * 3)
n2_b = (3 * 2)
n2 = n2_a + n2_b
N_weights = n1 + n2
w = np.random.randn(N_weights)  

es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, es.result.fbest, sep=',', file=open("long_swimmer_weights.txt", "a"))


