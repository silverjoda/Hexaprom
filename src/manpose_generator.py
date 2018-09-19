import gym
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

env = gym.make("FetchPickAndPlace-v1")
s = env.reset()

# grip_pos, robot_qpos, object_pos

N_trn = 1000
N_tst = 100

trn_obs = []
tst_obs = []

trn_ctr = 0
while True:
    s, _, _, _ = env.step(env.action_space.sample())
    img = env.render(mode='rgb_array')
    plt.imshow(img)
    plt.show()
    if np.random.rand() < 0.1:
        trn_obs.append(s)
        trn_ctr += 1
    if trn_ctr == N_trn:
        break

tst_ctr = 0
s = env.reset()
while True:
    s, _, _, _ = env.step(env.action_space.sample())
    if np.random.rand() < 0.05:
        tst_obs.append(s)
        tst_ctr += 1
    if trn_ctr == N_tst:
        break














