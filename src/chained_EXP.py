import numpy as np
import gym
import cma
from weight_distributor import Wdist
from time import sleep
import quaternion

# the function we want to optimize
def f(w):

    reward = 0
    done = False
    env_obs = env.reset()
    goal_dir = 0 #np.random.rand() * 3.14 * 2 - 3.14

    joint_activation_list = []
    master_activation_list = []

    while not done:

        # Observations
        # x,y,th : 0:3
        # j1,j2,j3,j4... : 3:21
        # x,y,th : 21:24

        torques = []
        newstates = []

        #obs = list(env_obs[2:3]) + states[0:1]
        obs = [0] + states[0:1]
        l1_m = np.tanh(np.matmul(obs, wdist.get_w('w_m1', w)) + wdist.get_w('b_m1', w))
        mout = np.tanh(np.matmul(l1_m, wdist.get_w('w_m2', w)) + wdist.get_w('b_m2', w))

        master_activation_list.append((l1_m, mout))

        for i in range(N):
            if i == 0:
                nt = [mout] + states[1:2]
            elif i == N - 1:
                nt = states[N-1:N] + [0]
            else:
                nt = states[i-1:i] + states[i+1:i+2]
            obs = list(env_obs[i+3:i+4]) + nt
            l1 = np.tanh(np.matmul(obs, wdist.get_w('w_l1', w)) + wdist.get_w('b_l1', w))
            t, s = np.tanh(np.matmul(l1, wdist.get_w('w_l2', w)) + wdist.get_w('b_l2', w))
            torques.append(t)
            newstates.append(s)

            joint_activation_list.append((l1,t,s))

        states = newstates
        # -------------

        # Step environment
        env_obs, rew, done, _ = env.step(torques)

        xposafter = env_obs[0]
        yposafter = env_obs[1]

        reward_dir = np.cos(goal_dir) * (xposafter - xposbefore) + np.sin(goal_dir) * (yposafter - yposbefore)
        reward_dir /= 0.04
        xposbefore = xposafter
        yposbefore = yposafter

        if animate:
            env.render()

        reward += rew

    l1_list, t_list, s_list = zip(*joint_activation_list)
    l1_mean, l1_std = np.mean(l1_list), np.std(l1_list)
    t_mean, t_std = np.mean(t_list), np.std(t_list)
    s_mean, s_std = np.mean(s_list), np.std(s_list)

    l1_m_list, mout_list = zip(*master_activation_list)
    mout_mean, mout_std = np.mean(mout_list), np.std(mout_list)
    l1_m_mean, l1_m_std = np.mean(l1_m_list), np.std(l1_m_list)

    # Print run statistics
    print("w_mean: {}, w_std: {}, w_min: {}, w_max: {}, m_l1_mean: {}, m_l1_std: {}, m_out_mean: {}, m_out_std: {}, l1_mean: {}, l1_std: {}, s_mean: {}, s_std: {}, t_mean: {}, t_std: {}".
          format(np.mean(w), np.std(w), np.min(w), np.max(w), l1_m_mean, l1_m_std, mout_mean, mout_std, l1_mean, l1_std, s_mean, s_std, t_mean, t_std))

    return -reward

# Make environment
env = gym.make("Snek-v0")
print("Action space: {}, observation space: {}".format(env.action_space.shape, env.observation_space.shape))
animate = True

N = env.action_space.shape[0]

# Generate weights
wdist = Wdist()

afun = np.tanh
actfun = lambda x:x

print("afun: {}".format(afun))

wdist.addW((3, 3), 'w_l1')
wdist.addW((3,), 'b_l1')

wdist.addW((3, 2), 'w_l2')
wdist.addW((2,), 'b_l2')

wdist.addW((2, 2), 'w_m1')
wdist.addW((2,), 'b_m1')

wdist.addW((2, 1), 'w_m2')
wdist.addW((1,), 'b_m2')

N_weights = wdist.get_N()
print("Nweights: {}".format(N_weights))
w = np.random.randn(N_weights) * 0.5

es = cma.CMAEvolutionStrategy(w, 0.5)

try:
    es.optimize(f, iterations=2000)
except KeyboardInterrupt:
    print("User interrupted process.")
es.result_pretty()

print(es.result.xbest, es.result.fbest, sep=',', file=open("snek_weights.txt", "a"))


