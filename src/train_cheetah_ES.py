import numpy as np
import gym
import cma
np.random.seed(0)

def ff(w):

    reward = 0
    step = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0,0,0,0]

    while not done:

        # Make actions:

        # Oscillator 0
        o0 = list(env_obs) + prev_torques
        a0, m0 = np.tanh(np.matmul(o0, w[0:n1].reshape((23,2))))

        # Oscillator 1
        o1 = list(env_obs) + prev_torques
        a1, m1 = np.tanh(np.matmul(o1, w[n1:n1 + n2].reshape((23,2))))

        # Oscillator 2
        o2 = list(env_obs) + prev_torques
        a2, m2 = np.tanh(np.matmul(o2, w[n2:n2 + n3].reshape((23, 2))))

        # Oscillator 3
        o3 = list(env_obs) + prev_torques
        a3, m3 = np.tanh(np.matmul(o3, w[n3:n3 + n4].reshape((23, 2))))

        # Oscillator 4
        o4 = list(env_obs) + prev_torques
        a4, m4 = np.tanh(np.matmul(o4, w[n4:n4 + n5].reshape((23, 2))))

        # Oscillator 5.
        o5 = list(env_obs) + prev_torques
        a5, m5 = np.tanh(np.matmul(o5, w[n5:n5 + n6].reshape((23, 2))))

        mult = 1
        offset = 0

        t0 = (offset + mult * a0) * np.sin(m0 * step)
        t1 = (offset + mult * a1) * np.sin(m1 * step)
        t2 = (offset + mult * a2) * np.sin(m2 * step)
        t3 = (offset + mult * a3) * np.sin(m3 * step)
        t4 = (offset + mult * a4) * np.sin(m4 * step)
        t5 = (offset + mult * a5) * np.sin(m5 * step)

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t1, t2, t3, t4, t5])

        # TODO: Inspect oscillator, action values etc
        # TODO: Try optimize normal mlp policy but predicting oscillator values instead of torques

        if animate:
            env.render()

        reward += rew
        step += 0.03

        prev_torques = [t0, t1, t2, t3, t4, t5]

    return -reward

# the function we want to optimize
def f(w):

    reward = 0
    step = 0
    done = False
    env_obs = env.reset()
    prev_torques = [0,0,0,0,0,0]

    while not done:

        # Env obs:
        # 0,  1,    2,    3,    4,    5,    6,    7    8,  9,  10,    11,    12,    13,    14,    15,    16
        # z, th, ll_0, ll_1, ll_2, rl_0, rl_1, rl_2 - dx, dz, dth, dll_0, dll_1, dll_2, drl_0, drl_1, drl_2

        # Make actions:

        # Oscillator 0
        o0 = list(env_obs[[0, 1, 2, 5, 8, 9, 11, 14]]) + list(np.array(prev_torques)[[0,1,2,3]])
        a0, m0 = np.tanh(np.matmul(o0, w[0:n1].reshape((12, 2))))

        # Oscillator 1
        o1 = list(env_obs[[0, 1, 2, 5, 8, 9, 11, 14]]) + list(np.array(prev_torques)[[0,1,2,3]])
        a1, m1 = np.tanh(np.matmul(o1, w[n1:n1 + n2].reshape((12, 2))))

        # Oscillator 2
        o2 = list(env_obs[[2,3,4,11,12,13]]) + list(np.array(prev_torques)[[0,2,4]])
        a2, m2 = np.tanh(np.matmul(o2, w[n2:n2 + n3].reshape((9, 2))))

        # Oscillator 3
        o3 = list(env_obs[[5,6,7,14,15,16]]) + list(np.array(prev_torques)[[1,3,5]])
        a3, m3 = np.tanh(np.matmul(o3, w[n3:n3 + n4].reshape((9, 2))))

        # Oscillator 4
        o4 = list(env_obs[[3,4,15,16]]) + list(np.array(prev_torques)[[2,4]])
        a4, m4 = np.tanh(np.matmul(o4, w[n4:n4 + n5].reshape((6, 2))))

        # Oscillator 5.
        o5 = list(env_obs[[6,7,15,16]]) + list(np.array(prev_torques)[[3,5]])
        a5, m5 = np.tanh(np.matmul(o5, w[n5:n5 + n6].reshape((6, 2))))

        mult = 1
        offset = 0

        #print([m0, m1, m2, m3, m4, m5])


        t0 = (offset + mult * a0) * np.sin(m0 * step)
        t1 = (offset + mult * a1) * np.sin(m1 * step)
        t2 = (offset + mult * a2) * np.sin(m2 * step)
        t3 = (offset + mult * a3) * np.sin(m3 * step)
        t4 = (offset + mult * a4) * np.sin(m4 * step)
        t5 = (offset + mult * a5) * np.sin(m5 * step)

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t2, t4, t1, t3, t5])

        if animate:
            env.render()

        reward += rew
        step += 0.01

        # TODO: Try to add PD controller

        prev_torques = [t0, t1, t2, t3, t4, t5]

    return -reward

# Make environment
env = gym.make("HalfCheetah-v2")
animate = True

# # Generate weights
# n1 = (12 * 2 + 0)
# n2 = (12 * 2)
# n3 = (9 * 2)
# n4 = (9 * 2)
# n5 = (6 * 2)
# n6 = (6 * 2)

# Generate weights
n1 = (23 * 2)
n2 = (23 * 2)
n3 = (23 * 2)
n4 = (23 * 2)
n5 = (23 * 2)
n6 = (23 * 2)

N_weights = n1 + n2 + n3 + n4 + n5 + n6

# TODO: Try again full obvservation policy but with correct weight scaling and maybe PID
W_MULT = 0.05
w = np.random.randn(N_weights) * W_MULT

print("N_weights: {}".format(N_weights))

es = cma.CMAEvolutionStrategy(w, 0.5)
es.optimize(ff, iterations=10000)
es.result_pretty()

print(es.result.xbest, file=open("halfcheetah_weights.txt", "a"))
