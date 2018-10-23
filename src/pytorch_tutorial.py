import argparse
from collections import deque
import copy
from functools import partial
import gc
import logging
from multiprocessing.pool import ThreadPool
import os
import pickle
import random
import sys
import time

# from evostra import EvolutionStrategy
from pytorch_es import EvolutionModule
from pytorch_es.utils.helpers import weights_init
import gym
from gym import logger as gym_logger
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn

gym_logger.setLevel(logging.CRITICAL)


# add the model on top of the convolutional base
model = nn.Sequential(
    nn.Linear(42, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, 17)
)

model.apply(weights_init)


def get_reward(weights, model, render=False):
    cloned_model = copy.deepcopy(model)
    for i, param in enumerate(cloned_model.parameters()):
        try:
            param.data.copy_(weights[i])
        except:
            param.data.copy_(weights[i].data)

    ob = env.reset()
    done = False
    total_reward = 0
    while not done:
        if render:
            env.render()
        batch = torch.from_numpy(ob[np.newaxis, ...]).float()

        prediction = cloned_model(Variable(batch, volatile=True))
        action = prediction.data[0]
        ob, reward, done, _ = env.step(action)

        total_reward += reward

    return total_reward

env = gym.make("Snek-v0")

partial_func = partial(get_reward, model=model)
mother_parameters = list(model.parameters())

es = EvolutionModule(
    mother_parameters, partial_func, population_size=50,
    sigma=0.1, learning_rate=0.001, reward_goal=300, consecutive_goal_stopping=20,
    threadcount=8, cuda=False, render_test=True
)
start = time.time()
final_weights = es.run(4000, print_step=1)
end = time.time() - start

pickle.dump(final_weights, open(os.path.abspath("agents/ES"), 'wb'))

reward = partial_func(final_weights, render=True)

print(f"Reward from final weights: {reward}")
print(f"Time to completion: {end}")
