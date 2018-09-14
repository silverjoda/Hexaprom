import numpy as np
import gym
import cma
from pid import PDreg
from weight_distributor import Wdist
np.random.seed(0)
from time import sleep
import quaternion

def ikt(j):
    pass

def f(w):

    reward = 0
    done = False
    env_obs = env.reset()

    # Legs: fl, rl, rr, fr

    # Observations
    # 0,  1,  2,  3,  4,   5,   6,   7,   8,   9,  10,  11,  12, 13, 14, 15,  16,     17,   18,    19,   20,   21,   22,   23,   24,   25,   26
    # z, q1, q2, q3, q4, l0f, l0c, l1f, l1c, l2f, l2c, l3f, l3c, dx, dy, dz,  dthx, dthy, dthz,  dl0f, dl0c, dl1f, dl1c, dl2f, dl2c, dl3f, dl3c

    while not done:

        h0, f0, h1, f1, h2, f2, h3, f3 = env_obs[5:13]
        _,_,z = quaternion.as_euler_angles(np.quaternion(*env_obs[1:5]))
        z = np.expand_dims(z, 0)

        # fl
        l1 = np.tanh(np.matmul([h0, f0, h1, f1, h2, f2, h3, f3, z], wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w))
        l2 = np.tanh(np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w))
        joints_ref = np.tanh(np.matmul(l2, wdist.get_w('w_l3', w)) + wdist.get_w('b_l3', w))

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

# Master node
wdist.addW((9, 6), 'w_l1')
wdist.addW((6,), 'b_l1')

wdist.addW((4, 6), 'w_l2')
wdist.addW((6,), 'b_l2')

wdist.addW((6, 8), 'w_l3')
wdist.addW((8,), 'b_l3')


N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 0.1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT
es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=10000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("ant_weights.txt", "a"))
