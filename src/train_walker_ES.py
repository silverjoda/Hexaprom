import numpy as np
import gym
import cma
from pid import PDreg
np.random.seed(0)
from weight_distributor import Wdist


# EXP:  CLone experiment and add control heirarchy (first policy controls femurs, femurs control coxas)

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    rnn_states = {'h_m' : [0,0,0,0], 'h_h1' : [0,0], 'h_k1' : [0,0],'h_f1' : [0,0], 'h_h2' : [0,0], 'h_k2' : [0,0],'h_f2' : [0,0]}

    # Observations
    # 0,  1,   2,   3,   4,   5,   6,   7,  8,  9,  10,   11,   12,   13,   14,   15,   16
    # z, th, lj0, lj1, lj2, rj0, rj1, rj2, dx, dz, dth, dlj0, dlj1, dlj2, drj0, drj1, drj2

    while not done:

        # Master node
        obs = list(env_obs[[0, 1, 8, 9, 10]]) + list(rnn_states['h_h1']) + list(rnn_states['h_h2'])
        h_m = np.tanh(np.matmul(wdist.get_w('m_w', w).T, rnn_states['h_m']) + np.matmul(wdist.get_w('m_u', w).T, obs) + wdist.get_w('m_b', w))
        y_m = np.matmul(h_m, wdist.get_w('m_v', w)) + wdist.get_w('m_c', w)

        # h1 node
        obs = list(env_obs[2:3]) + list(y_m[0:2]) + list(rnn_states['h_k1'])
        h_h1 = np.tanh(np.matmul(wdist.get_w('h_w', w).T, rnn_states['h_h1']) + np.matmul(wdist.get_w('h_u', w).T, obs) + wdist.get_w('h_b', w))
        y_h1 = np.matmul(wdist.get_w('h_v', w).T, h_h1) + wdist.get_w('h_c', w)

        # h2 node
        obs = list(env_obs[5:6]) + list(y_m[2:4]) + list(rnn_states['h_k2'])
        h_h2 = np.tanh(np.matmul(wdist.get_w('h_w', w).T, rnn_states['h_h2']) + np.matmul(wdist.get_w('h_u', w).T,obs) + wdist.get_w('h_b', w))
        y_h2 = np.matmul(wdist.get_w('h_v', w).T, h_h2) + wdist.get_w('h_c', w)

        # k1 node
        obs = list(env_obs[3:4]) + list(rnn_states['h_h1']) + list(rnn_states['h_f1'])
        h_k1 = np.tanh(np.matmul(wdist.get_w('k_w', w).T, rnn_states['h_k1']) + np.matmul(wdist.get_w('k_u', w).T, obs) + wdist.get_w('k_b', w))
        y_k1 = np.matmul(wdist.get_w('k_v', w).T, h_k1) + wdist.get_w('k_c', w)

        # k2 node
        obs = list(env_obs[6:7]) + list(rnn_states['h_h2']) + list(rnn_states['h_f2'])
        h_k2 = np.tanh(np.matmul(wdist.get_w('k_w', w).T, rnn_states['h_k2']) + np.matmul(wdist.get_w('k_u', w).T,obs) + wdist.get_w('k_b', w))
        y_k2 = np.matmul(wdist.get_w('k_v', w).T, h_k2) + wdist.get_w('k_c', w)

        # f1 node
        obs = list(env_obs[4:5]) + list(rnn_states['h_k1'])
        h_f1 = np.tanh(np.matmul(wdist.get_w('f_w', w).T, rnn_states['h_f1']) + np.matmul(wdist.get_w('f_u', w).T, obs) + wdist.get_w('f_b', w))
        y_f1 = np.matmul(wdist.get_w('f_v', w).T, h_f1) + wdist.get_w('f_c', w)

        # f1 node
        obs = list(env_obs[7:8]) + list(rnn_states['h_k2'])
        h_f2 = np.tanh(np.matmul(wdist.get_w('f_w', w).T, rnn_states['h_f2']) + np.matmul(wdist.get_w('f_u', w).T, obs) + wdist.get_w('f_b', w))
        y_f2 = np.matmul(wdist.get_w('f_v', w).T, h_f2) + wdist.get_w('f_c', w)

        rnn_states = {'h_m': h_m, 'h_h1': h_h1, 'h_k1': h_k1, 'h_f1': h_f1,
                      'h_h2': h_h2, 'h_k2': h_k2, 'h_f2': h_f2}

        action = [y_h1, y_k1, y_f1, y_h2, y_k2, y_f2]

        # Step environment
        env_obs, rew, done, _ = env.step(action)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("Walker2d-v2")
animate = False

# Generate weights
wdist = Wdist()

# Master node
wdist.addW((4, 4), 'm_w') # Hidden -> Hidden
wdist.addW((9, 4), 'm_u') # Input -> Hidden
wdist.addW((4, 4), 'm_v') # Hidden -> Output
wdist.addW((4,), 'm_b') # Hidden bias
wdist.addW((4,), 'm_c') # Output bias

# Hip node
wdist.addW((2, 2), 'h_w')
wdist.addW((5, 2), 'h_u')
wdist.addW((2, 1), 'h_v')
wdist.addW((2,), 'h_b')
wdist.addW((1,), 'h_c')

# Knee node
wdist.addW((2, 2), 'k_w')
wdist.addW((5, 2), 'k_u')
wdist.addW((2, 1), 'k_v')
wdist.addW((2,), 'k_b')
wdist.addW((1,), 'k_c')

# Foot node
wdist.addW((2, 2), 'f_w')
wdist.addW((3, 2), 'f_u')
wdist.addW((2, 1), 'f_v')
wdist.addW((2,), 'f_b')
wdist.addW((1,), 'f_c')


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


print(es.result.xbest, file=open("walker_weights.txt", "a"))
