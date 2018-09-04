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
    rnn_states = {'h_m' : [0] * n_hidden_m,
                  'h_h1' : [0] * n_hidden_osc,
                  'h_k1' : [0] * n_hidden_osc,
                  'h_f1' : [0] * n_hidden_osc,
                  'h_h2' : [0] * n_hidden_osc,
                  'h_k2' : [0] * n_hidden_osc,
                  'h_f2' : [0] * n_hidden_osc}

    # Observations
    # 0,  1,   2,   3,   4,   5,   6,   7,  8,  9,  10,   11,   12,   13,   14,   15,   16
    # z, th, lj0, lj1, lj2, rj0, rj1, rj2, dx, dz, dth, dlj0, dlj1, dlj2, drj0, drj1, drj2

    pdreg = PDreg(3, 0.05)

    while not done:

        # Master node
        obs = list(env_obs[[0, 1, 8, 9, 10]]) + list(rnn_states['h_h1']) + list(rnn_states['h_h2'])
        h_m = np.tanh(np.matmul(wdist.get_w('m_w', w).T, rnn_states['h_m']) + np.matmul(wdist.get_w('m_u', w).T, obs) + wdist.get_w('m_b', w))
        y_m = np.matmul(h_m, wdist.get_w('m_v', w)) + wdist.get_w('m_c', w)

        # h1 node
        obs = list(env_obs[2:3]) + list(y_m[0:n_hidden_osc]) + list(rnn_states['h_k1'])
        h_h1 = np.tanh(np.matmul(wdist.get_w('h_w', w).T, rnn_states['h_h1']) + np.matmul(wdist.get_w('h_u', w).T, obs) + wdist.get_w('h_b', w))
        y_h1 = np.matmul(wdist.get_w('h_v', w).T, h_h1) + wdist.get_w('h_c', w)

        # h2 node
        obs = list(env_obs[5:6]) + list(y_m[n_hidden_osc:]) + list(rnn_states['h_k2'])
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
        #pd_action = pdreg.update([a[0] for a in action], env_obs[2:8])

        # Step environment
        env_obs, rew, done, _ = env.step(action)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("Walker2d-v2")
animate = True

# Generate weights
wdist = Wdist()

n_hidden_m = 4
n_hidden_osc = 2

# Master node
wdist.addW((n_hidden_m, n_hidden_m), 'm_w') # Hidden -> Hidden
wdist.addW((5 + 2 * n_hidden_osc, n_hidden_m), 'm_u') # Input -> Hidden
wdist.addW((n_hidden_m, 2 * n_hidden_osc), 'm_v') # Hidden -> Output
wdist.addW((n_hidden_m,), 'm_b') # Hidden bias
wdist.addW((2 * n_hidden_osc,), 'm_c') # Output bias

# Hip node
wdist.addW((n_hidden_osc, n_hidden_osc), 'h_w')
wdist.addW((1 + 2 * n_hidden_osc, n_hidden_osc), 'h_u')
wdist.addW((n_hidden_osc, 1), 'h_v')
wdist.addW((n_hidden_osc,), 'h_b')
wdist.addW((1,), 'h_c')

# Knee node
wdist.addW((n_hidden_osc, n_hidden_osc), 'k_w')
wdist.addW((1 + 2 * n_hidden_osc, n_hidden_osc), 'k_u')
wdist.addW((n_hidden_osc, 1), 'k_v')
wdist.addW((n_hidden_osc,), 'k_b')
wdist.addW((1,), 'k_c')

# Foot node
wdist.addW((n_hidden_osc, n_hidden_osc), 'f_w')
wdist.addW((1 + n_hidden_osc, n_hidden_osc), 'f_u')
wdist.addW((n_hidden_osc, 1), 'f_v')
wdist.addW((n_hidden_osc,), 'f_b')
wdist.addW((1,), 'f_c')

N_weights = wdist.get_N()
print("Nweights: {}, nhm: {}, nho: {}".format(N_weights, n_hidden_m, n_hidden_osc))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print('w = [' + ','.join(map(str, es.result.xbest)) + '] '  + str(es.result.fbest), file=open("walker_weights.txt", "a"))
