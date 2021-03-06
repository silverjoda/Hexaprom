import numpy as np
import gym
import cma
from pid import PDreg
from weight_distributor import Wdist
np.random.seed(0)
from time import sleep
# EXP: Use as state just current end-points of legs and use IK to calculate joints. NO inputs are taken from OBS

def ikt(j):
    pass

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    eef_pos = 4 * [0]

    pdreg = PDreg(2.0, 0.005)

    # Legs: fl, rl, rr, fr

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # fl
        obs = list(np.array(eef_pos)[[0, 1, 3]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wfl1', w)) + wdist.get_w('bfl1', w))
        fl_ref = np.matmul(l, wdist.get_w('wfl2', w)) + wdist.get_w('bfl2', w)

        # fr
        obs = list(np.array(eef_pos)[[3, 2, 0]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wfr1', w)) + wdist.get_w('bfr1', w))
        fr_ref = np.matmul(l, wdist.get_w('wfr2', w)) + wdist.get_w('bfr2', w)

        # rl
        obs = list(np.array(eef_pos)[[1, 0, 2]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wrl1', w)) + wdist.get_w('brl1', w))
        rl_ref = np.matmul(l, wdist.get_w('wrl2', w)) + wdist.get_w('brl2', w)

        # rr
        obs = list(np.array(eef_pos)[[2, 3, 1]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wrr1', w)) + wdist.get_w('brr1', w))
        rr_ref = np.matmul(l, wdist.get_w('wrr2', w)) + wdist.get_w('brr2', w)

        joints_ref = ikt(fl_ref) + ikt(rl_ref) + ikt(rr_ref) + ikt(fr_ref)
        joints_action = pdreg.update(env_obs[5:13], joints_ref)

        # Step environment
        env_obs, rew, done, _ = env.step(joints_action)

        if animate:
            env.render()

        reward += rew
        eef_pos = joints_ref

    return -reward

# Make environment
env = gym.make("Ant-v3")
animate = False

# Generate weights
wdist = Wdist()
wdist.addW((4, 4), 'wfl1')
wdist.addW((4,), 'bfl1')
wdist.addW((4, 1), 'wfl2')
wdist.addW((1,), 'bfl2')

wdist.addW((4, 4), 'wrl1')
wdist.addW((4,), 'bflr')
wdist.addW((4, 1), 'wfr2')
wdist.addW((1,), 'bfr2')

wdist.addW((4, 4), 'wrr1')
wdist.addW((4,), 'brr1')
wdist.addW((4, 1), 'wrr2')
wdist.addW((1,), 'brr2')

wdist.addW((4, 4), 'wfr1')
wdist.addW((4,), 'bfr1')
wdist.addW((4, 1), 'wfr2')
wdist.addW((1,), 'bfr2')

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
