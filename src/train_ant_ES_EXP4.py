import numpy as np
import gym
import cma
from pid import PDreg
from weight_distributor import Wdist
np.random.seed(0)
from time import sleep
# EXP: Micro policies control 1 leg each, inputs are only other leg states. NO inputs are taken from OBS

def ikt(j):
    pass

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    s = {'fl': [np.random.randn()], 'rl': [np.random.randn()], 'rr': [np.random.randn()], 'fr': [np.random.randn()]}

    pdreg = PDreg(2.0, 0.005)

    # Legs: fl, rl, rr, fr

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # fl
        obs = s['fl'] + s['rl'] + s['fr'] + list(env_obs[[5,6]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wfl1', w)) + wdist.get_w('bfl1', w))
        fl_ref_a, fl_ref_b, sfl = np.tanh(np.matmul(l, wdist.get_w('wfl2', w)) + wdist.get_w('bfl2', w))

        # rl
        obs = s['rl'] + s['fl'] + s['rr'] + list(env_obs[[7,8]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wrl1', w)) + wdist.get_w('brl1', w))
        rl_ref_a, rl_ref_b, srl = np.tanh(np.matmul(l, wdist.get_w('wrl2', w)) + wdist.get_w('brl2', w))

        # rr
        obs = s['rr'] + s['fr'] + s['rl'] + list(env_obs[[9,10]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wrr1', w)) + wdist.get_w('brr1', w))
        rr_ref_a, rr_ref_b, srr = np.tanh(np.matmul(l, wdist.get_w('wrr2', w)) + wdist.get_w('brr2', w))

        # fr
        obs = s['fr'] + s['rr'] + s['fl'] + list(env_obs[[11, 12]])
        l = np.tanh(np.matmul(obs, wdist.get_w('wfr1', w)) + wdist.get_w('bfr1', w))
        fr_ref_a, fr_ref_b, sfr = np.tanh(np.matmul(l, wdist.get_w('wfr2', w)) + wdist.get_w('bfr2', w))

        joints_ref = [fl_ref_a, fl_ref_b, rl_ref_a, rl_ref_b, rr_ref_a, rr_ref_b, fr_ref_a, fr_ref_b]
        joints_action = pdreg.update(env_obs[5:13], joints_ref)

        s = {'fl': [sfl], 'rl': [srl], 'rr': [srr], 'fr':[sfr]}

        # Step environment
        env_obs, rew, done, _ = env.step(joints_ref)

        if animate:
            env.render()

        reward += rew

    return -reward

# Make environment
env = gym.make("Ant-v3")
animate = False

# Generate weights
wdist = Wdist()
wdist.addW((5, 5), 'wfl1')
wdist.addW((5,), 'bfl1')
wdist.addW((5, 3), 'wfl2')
wdist.addW((3,), 'bfl2')

wdist.addW((5, 5), 'wrl1')
wdist.addW((5,), 'brl1')
wdist.addW((5, 3), 'wrl2')
wdist.addW((3,), 'brl2')

wdist.addW((5, 5), 'wrr1')
wdist.addW((5,), 'brr1')
wdist.addW((5, 3), 'wrr2')
wdist.addW((3,), 'brr2')

wdist.addW((5, 5), 'wfr1')
wdist.addW((5,), 'bfr1')
wdist.addW((5, 3), 'wfr2')
wdist.addW((3,), 'bfr2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 0.3
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("ant_weights.txt", "a"))
