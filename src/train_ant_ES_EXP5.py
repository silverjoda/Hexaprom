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
    s = {'fl': [np.random.randn()], 'rl': [np.random.randn()], 'rr': [np.random.randn()], 'fr': [np.random.randn()]}

    pdreg = PDreg(2.0, 0.005)

    # Legs: fl, rl, rr, fr

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        # fl
        obs = s['fl'] + s['rl'] + s['fr'] + list(env_obs[[5,6,19,20,13]]) + [1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        fl_ref_a, fl_ref_b, sfl = np.tanh(np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w))

        # rl
        obs = s['rl'] + s['fl'] + s['rr'] + list(env_obs[[7,8,21,22,13]]) + [-1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        rl_ref_a, rl_ref_b, srl = np.tanh(np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w))

        # rr
        obs = s['rr'] + s['fr'] + s['rl'] + list(env_obs[[9,10,23,24,13]]) + [-1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        rr_ref_a, rr_ref_b, srr = np.tanh(np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w))

        # fr
        obs = s['fr'] + s['rr'] + s['fl'] + list(env_obs[[11,12,25,26,13]]) + [1]
        l = np.tanh(np.matmul(obs, wdist.get_w('w1', w)) + wdist.get_w('b1', w))
        fr_ref_a, fr_ref_b, sfr = np.tanh(np.matmul(l, wdist.get_w('w2', w)) + wdist.get_w('b2', w))

        joints_ref = [fl_ref_a, fl_ref_b, rl_ref_a, rl_ref_b, rr_ref_a, rr_ref_b, fr_ref_a, fr_ref_b]
        joints_ref = [j + np.random.rand() * 0.03 for j in joints_ref]
        #joints_action = pdreg.update(env_obs[5:13], joints_ref)

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
wdist.addW((9, 7), 'w1')
wdist.addW((7,), 'b1')
wdist.addW((7, 3), 'w2')
wdist.addW((3,), 'b2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT
#
# w=[ 4.04639843e+00,  1.69130766e+00, -1.54266978e-01,  2.00324220e+00,
#   2.67787942e+00, -1.40204274e+00,  1.98006357e+00,  9.99608999e-02,
#   1.22520323e+00, -5.37618420e-01,  1.00650941e+00,  2.24156756e+00,
#   8.54904993e-01,  6.93011918e-01,  1.47490245e+00, -2.70949929e+00,
#   1.45077113e+00, -4.93592249e-01,  2.88642284e-01,  4.07439413e-01,
#  -2.05303680e+00, -2.24487143e+00, -7.16711311e-01,  1.23589555e+00,
#   5.05095272e+00,  9.40011848e-01, -3.66496690e+00, -5.79538862e+00,
#   2.09924641e+00, -5.32341724e-01,  1.08849939e+00,  4.60101805e-01,
#   3.79369408e-01, -1.55319768e+00, -3.00919418e+00,  1.37863858e+00,
#  -3.36374685e+00,  3.97263691e+00, -3.21101317e-01, -3.02172890e+00,
#  -8.51034566e-01, -3.65180300e+00, -2.69824439e+00,  4.03876672e+00,
#   1.25491085e+00,  3.55007634e+00, -3.07983672e+00,  6.60509578e+00,
#   2.90655888e+00, -9.54683452e-01,  2.76247740e-01,  2.12828931e+00,
#   1.17591603e+00, -1.56775435e+00, -2.60167141e+00,  3.75955808e+00,
#  -2.17082826e+00,  2.20755686e-01, -6.21599439e-01,  2.52313892e-01,
#   1.13622893e+00, -9.96868432e-01,  8.46255191e-01, -1.68303591e-02,
#   3.74266965e+00, -1.55941434e+00, -1.53468062e+00, -1.46300624e+00,
#   9.40761112e-01, -7.92554776e-01, -3.53622215e-02,  6.90852678e-02,
#  -1.41143688e+00, -2.31318110e-01, -2.51176809e-01, -1.52768392e+00,
#   5.59499679e-03, -1.99584377e-01,  1.78720545e+00,  1.40801347e-01,
#   1.81101693e-02, -3.65495994e+00,  2.21374042e-02,  1.01901985e-01,
#   1.95054332e+00, -4.46407244e-02,  5.25472644e-01,  3.14240328e+00,
#   1.57598006e-01,  1.10481523e-01,  7.54556497e-02,  1.22380166e-01,
#   1.20460508e-01,  3.43380314e+00]


es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("ant_weights.txt", "a"))
