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

def relu(x):
    return np.maximum(x,0)

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
        h0_up = afun(np.matmul([h0, f0], wdist.get_w('w_up', w)) + wdist.get_w('b_up', w))
        h1_up = afun(np.matmul([h1, f1], wdist.get_w('w_up', w)) + wdist.get_w('b_up', w))
        h2_up = afun(np.matmul([h2, f2], wdist.get_w('w_up', w)) + wdist.get_w('b_up', w))
        h3_up = afun(np.matmul([h3, f3], wdist.get_w('w_up', w)) + wdist.get_w('b_up', w))

        m_down = afun(np.matmul(np.concatenate([h0_up, h1_up, h2_up, h3_up, z], 0), wdist.get_w('w_m_down1', w)) + wdist.get_w('b_m_down1', w))
        m_down = afun(np.matmul(m_down, wdist.get_w('w_m_down2', w)) + wdist.get_w('b_m_down2', w))

        h0_down = afun(np.matmul(m_down[0:2], wdist.get_w('w_h_down', w)) + wdist.get_w('b_h_down', w))
        h1_down = afun(np.matmul(m_down[2:4], wdist.get_w('w_h_down', w)) + wdist.get_w('b_h_down', w))
        h2_down = afun(np.matmul(m_down[4:6], wdist.get_w('w_h_down', w)) + wdist.get_w('b_h_down', w))
        h3_down = afun(np.matmul(m_down[6:8], wdist.get_w('w_h_down', w)) + wdist.get_w('b_h_down', w))

        h0_sig = np.concatenate([m_down[0:2], [h0]],0)
        h1_sig = np.concatenate([m_down[2:4], [h1]],0)
        h2_sig = np.concatenate([m_down[4:6], [h2]],0)
        h3_sig = np.concatenate([m_down[6:8], [h3]],0)

        f0_sig = np.concatenate([h0_down, [f0]])
        f1_sig = np.concatenate([h1_down, [f1]])
        f2_sig = np.concatenate([h2_down, [f2]])
        f3_sig = np.concatenate([h3_down, [f3]])

        h0_act = actfun(np.matmul(h0_sig, wdist.get_w('w_h_act', w)) + wdist.get_w('b_h_act', w))
        h1_act = actfun(np.matmul(h1_sig, wdist.get_w('w_h_act', w)) + wdist.get_w('b_h_act', w))
        h2_act = actfun(np.matmul(h2_sig, wdist.get_w('w_h_act', w)) + wdist.get_w('b_h_act', w))
        h3_act = actfun(np.matmul(h3_sig, wdist.get_w('w_h_act', w)) + wdist.get_w('b_h_act', w))

        f0_act = actfun(np.matmul(f0_sig, wdist.get_w('w_f_act', w)) + wdist.get_w('b_f_act', w))
        f1_act = actfun(np.matmul(f1_sig, wdist.get_w('w_f_act', w)) + wdist.get_w('b_f_act', w))
        f2_act = actfun(np.matmul(f2_sig, wdist.get_w('w_f_act', w)) + wdist.get_w('b_f_act', w))
        f3_act = actfun(np.matmul(f3_sig, wdist.get_w('w_f_act', w)) + wdist.get_w('b_f_act', w))

        joints_ref = [h0_act, f0_act, h1_act, f1_act, h2_act, f2_act, h3_act, f3_act]

        # Step environment
        env_obs, rew, done, _ = env.step(joints_ref)

        if animate:
            env.render()

        reward += rew

    return -reward


# Make environment
env = gym.make("Ant-v3")
animate = True

# Generate weights
wdist = Wdist()

afun = np.tanh
actfun = lambda x:x

print("afun: {}".format(afun))

# Master node
wdist.addW((2, 2), 'w_up')
wdist.addW((2,), 'b_up')

wdist.addW((9, 4), 'w_m_down1')
wdist.addW((4,), 'b_m_down1')
wdist.addW((4, 8), 'w_m_down2')
wdist.addW((8,), 'b_m_down2')

wdist.addW((2, 2), 'w_h_down')
wdist.addW((2,), 'b_h_down')

wdist.addW((3, 1), 'w_h_act')
wdist.addW((1,), 'b_h_act')

wdist.addW((3, 1), 'w_f_act')
wdist.addW((1,), 'b_f_act')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
W_MULT = 1
ACT_MULT = 1

w = np.random.randn(N_weights) * W_MULT

#w = np.asarray(w1)
# for i in range(10):
#     print(-f(w))
# exit()

es = cma.CMAEvolutionStrategy(w, 0.5)
try:
    es.optimize(f, iterations=4000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()


print(es.result.xbest, file=open("ant_weights.txt", "a"))
