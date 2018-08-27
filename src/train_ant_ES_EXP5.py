import numpy as np
import gym
import cma
from pid import PDreg
from weight_distributor import Wdist
np.random.seed(0)
from time import sleep
# EXP: Micro policies control 1 leg each, inputs are only other leg states. NO inputs are taken from OBS, FULL SHARED W

def ikt(j):
    pass

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    joints_ref = 8 * [0]
    s = {'fl': [np.random.randn()], 'rl': [np.random.randn()], 'rr': [np.random.randn()], 'fr': [np.random.randn()]}

    pdreg = PDreg(2.0, 0.005)

    # Legs: fl, rl, rr, fr

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # fl
        obs = list(np.array(joints_ref)[[0,1, 2,3, 6,7]]) + s['fl'] + [1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        fl_ref_a, fl_ref_b, sfl = np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        # fr
        obs = list(np.array(joints_ref)[[6,7, 4,5 ,0,1]]) + s['fr'] + [1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        fr_ref_a, fr_ref_b, sfr = np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        # rl
        obs = list(np.array(joints_ref)[[2,3, 0,1, 4,5]]) + s['rl'] + [-1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        rl_ref_a, rl_ref_b, srl = np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        # rr
        obs = list(np.array(joints_ref)[[4,5, 6,7, 2,3]]) + s['rr'] + [-1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        rr_ref_a, rr_ref_b, srr = np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w)

        joints_ref = [fl_ref_a, fl_ref_b, rl_ref_a, rl_ref_b, rr_ref_a, rr_ref_b, fr_ref_a, fr_ref_b]
        joints_action = pdreg.update(env_obs[5:13], joints_ref)

        s = {'fl': [sfl], 'rl': [srl], 'rr': [srr], 'fr':[sfr]}

        # Step environment
        env_obs, rew, done, _ = env.step(joints_action)

        if animate:
            env.render()

        reward += rew


    return -reward

# Make environment
env = gym.make("Ant-v3")
animate = True

# Generate weights
wdist = Wdist()
wdist.addW((8, 5), 'w1')
wdist.addW((5,), 'b1')
wdist.addW((5, 3), 'w2')
wdist.addW((3,), 'b2')

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


print(es.result.xbest, file=open("ant_weights.txt", "a"))
