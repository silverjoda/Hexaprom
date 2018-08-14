import numpy as np
import gym
import cma
import time
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

    # TODO: FInd out which leg is which

    while not done:

        # Env obs:
        # 0,  1,    2,    3,    4,    5,    6,    7    8,  9,  10,    11,    12,    13,    14,    15,    16
        # z, th, ll_0, ll_1, ll_2, rl_0, rl_1, rl_2 - dx, dz, dth, dll_0, dll_1, dll_2, drl_0, drl_1, drl_2

        # Torques: ll_0, ll_1, ll_2, rl_0, rl_1, rl_2

        # l0
        o0 = list(env_obs[[0,1,9,10,2,3,5,11,12,14]]) + list(np.array(prev_torques)[[0,1,3]])
        t0 = mult * np.tanh(np.matmul(o0, w[0:n1].reshape((13, 1))))

        # r0
        o1 = list(env_obs[[0,1,9,10,2,5,6,11,14,15]]) + list(np.array(prev_torques)[[0,3,4]])
        t1 = mult * np.tanh(np.matmul(o1, w[n1:n1 + n2].reshape((13, 1))))

        # l1
        o2 = list(env_obs[[2,3,4,6,10,11,13,15]]) + list(np.array(prev_torques)[[0,1,2,4]])
        t2 = mult * np.tanh(np.matmul(o2, w[n2:n2 + n3].reshape((12, 1))))

        # r1
        o3 = list(env_obs[[5,6,7,3,14,15,16,12]]) + list(np.array(prev_torques)[[3,4,5,1]])
        t3 = mult * np.tanh(np.matmul(o3, w[n3:n3 + n4].reshape((12, 1))))

        # l2
        o4 = list(env_obs[[3,4,7,12,13,16]]) + list(np.array(prev_torques)[[1,2,5]])
        t4 = mult * np.tanh(np.matmul(o4, w[n4:n4 + n5].reshape((9, 1))))

        # r2
        o5 = list(env_obs[[6,7,4,15,16,13]]) + list(np.array(prev_torques)[[4,5,2]])
        t5 = mult * np.tanh(np.matmul(o5, w[n5:n5 + n6].reshape((9, 1))))

        # Step environment
        env_obs, rew, done, _ = env.step([t0, t2, t4, t1, t3, t5])

        if animate:
            env.render()

        reward += rew
        step += 0.01

        prev_torques = [t0, t1, t2, t3, t4, t5]

    return -reward

# Make environment
env = gym.make("HalfCheetah-v2")
animate = False


# # Generate weights
n1 = (13 * 1)
n2 = (13 * 1)
n3 = (12 * 1)
n4 = (12 * 1)
n5 = (9 * 1)
n6 = (9 * 1)

N_weights = n1 + n2 + n3 + n4 + n5 + n6

# TODO: Try again full obvservation policy but with correct weight scaling and maybe PID
mult = 1
W_MULT = 0.5
w = np.random.randn(N_weights) * W_MULT

print("N_weights: {}, mult {}, n_mult {}".format(N_weights, mult, W_MULT))

es = cma.CMAEvolutionStrategy(w, 0.5)
es.optimize(f, iterations=10000)
es.result_pretty()

print(es.result.xbest, file=open("halfcheetah_weights.txt", "a"))

