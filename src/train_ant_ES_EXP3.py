import numpy as np
import gym
import cma
from pid import PDreg
from weight_distributor import Wdist
np.random.seed(0)

# EXP:  CLone experiment and add control heirarchy (first policy controls femurs, femurs control coxas)

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    prev_torques = 8 * [0]

    pdreg = PDreg(1.5, 0.005)

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # lj0
        obs = list(env_obs[[0, 1, 2, 5, 8, 9, 11]]) + list(np.array(prev_torques)[[0, 3]])
        l = np.tanh(np.matmul(obs, wdist.get_w('w11', w)) + wdist.get_w('b11', w))
        t0 = np.matmul(l, wdist.get_w('w12', w)) + wdist.get_w('b12', w)

        # lj1
        obs = list(env_obs[[2, 3, 4, 12]]) + list(np.array(prev_torques)[[1, 4]]) + [t0]
        l = np.tanh(np.matmul(obs, wdist.get_w('w21', w)) + wdist.get_w('b21', w))
        t1 = np.matmul(l, wdist.get_w('w22', w)) + wdist.get_w('b22', w)

        # lj2
        obs = list(env_obs[[2, 3, 4, 13]]) + list(np.array(prev_torques)[[2]]) + [t0, t1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w31', w)) + wdist.get_w('b31', w))
        t2 = np.matmul(l, wdist.get_w('w32', w)) + wdist.get_w('b32', w)

        # rj0
        obs = list(env_obs[[0, 1, 5, 2, 9, 8, 14]]) + list(np.array(prev_torques)[[3, 0]])
        l = np.tanh(np.matmul(obs, wdist.get_w('w11', w)) + wdist.get_w('b11', w))
        t3 = np.matmul(l, wdist.get_w('w12', w)) + wdist.get_w('b12', w)

        # rj1
        obs = list(env_obs[[5, 6, 7, 15]]) + list(np.array(prev_torques)[[4, 1]]) + [t3]
        l = np.tanh(np.matmul(obs, wdist.get_w('w21', w)) + wdist.get_w('b21', w))
        t4 = np.matmul(l, wdist.get_w('w22', w)) + wdist.get_w('b22', w)

        # rj2
        obs = list(env_obs[[5, 6, 7, 16]]) + list(np.array(prev_torques)[[5]]) + [t3, t4]
        l = np.tanh(np.matmul(obs, wdist.get_w('w31', w)) + wdist.get_w('b31', w))
        t5 = np.matmul(l, wdist.get_w('w32', w)) + wdist.get_w('b32', w)

        action = [t0[0], t1[0], t2[0], t3[0], t4[0], t5[0]]
        action = np.array(action) + np.random.randn(6) * 0.05

        pid_action = pdreg.update(env_obs[[2, 3, 4, 5, 6, 7]], action)

        # Step environment
        env_obs, rew, done, _ = env.step(pid_action)

        if animate:
            env.render()

        reward += rew
        prev_torques = action

    return -reward

# Make environment
env = gym.make("Walker2d-v2")
animate = False

# Generate weights
wdist = Wdist()
wdist.addW((9, 4), 'w11')
wdist.addW((4,), 'b11')
wdist.addW((4, 1), 'w12')
wdist.addW((1,), 'b12')

wdist.addW((7, 3), 'w21')
wdist.addW((3,), 'b21')
wdist.addW((3, 1), 'w22')
wdist.addW((1,), 'b22')

wdist.addW((7, 3), 'w31')
wdist.addW((3,), 'b31')
wdist.addW((3, 1), 'w32')
wdist.addW((1,), 'b32')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
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
