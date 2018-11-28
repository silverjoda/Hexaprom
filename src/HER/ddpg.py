"""
Deep Deterministic Policy Gradient agent
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

# Torch
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from HER.ant_goal_env import AntG

# Lib
import numpy as np
import random
from copy import deepcopy

import utils

from IPython import display
import os

# Files
from HER.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from HER.replaybuffer import Buffer
from HER.actorcritic_ddpg import Actor, Critic

# Hyperparameters
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
MINIBATCH_SIZE = 64
NUM_EPISODES = 100000
MU = 0
SIGMA = 0.2
BUFFER_SIZE = 5000000
DISCOUNT = 0.97
TAU = 0.001
WARMUP = 160
EPSILON = 1.0
EPSILON_DECAY = 1e-6


class DDPG:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)
        self.targetActor = deepcopy(Actor(self.env))
        self.targetCritic = deepcopy(Critic(self.env))
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.act_dim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.discount = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []


    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):

        acts = self.targetActor(nextStateBatch)
        Qvals = self.targetCritic(nextStateBatch, acts)

        y_i = []
        for r,t,q in zip(rewardBatch, terminalBatch, Qvals):
            if t:
                y_i.append(r)
            else:
                y_i.append(r + self.discount * q)
        y_i = T.stack(y_i)

        return y_i

    
    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + TAU*orgParam.data)

            
  
    def getMaxAction(self, s):
        """Inputs: Current state of the episode
            Returns the action which maximizes the Q-value of the current state-action pair"""

        #noise = (self.epsilon * Variable(torch.FloatTensor(self.noise()))).detach()
        noise = (self.epsilon * T.randn(self.act_dim)).detach()

        action = self.actor(s).detach()
        actionNoise = action + noise
        return actionNoise


    def train(self, animate=False):
        print('Starting training...')
        
        for i in range(NUM_EPISODES):
            obs = self.env.reset()
            self.noise.reset()
            done = False
            ep_reward = 0


            while not done:

                # Get action
                action = self.getMaxAction(utils.to_tensor(obs, True)).numpy()[0]

                # Step episode
                obs_new, r, done, _ = self.env.step(action)

                if animate:
                    self.env.render()

                ep_reward += r
                obs = obs_new

                # Add transition
                self.replayBuffer.append((obs, action, obs_new, r, done))

                # Training loop
                if len(self.replayBuffer) >= self.batchSize:

                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)

                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)

                    # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)

                    criticLoss.backward()
                    self.criticOptim.step()

                    # Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -T.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    actorLoss.backward()
                    self.actorOptim.step()

                    # Update Targets
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)

                    if i % 10 == 0 and done:
                        print("Episode {}/{}, ep reward: {}, actr loss: {}, critic loss: {}".format(i, NUM_EPISODES, ep_reward, actorLoss, criticLoss))

            self.epsilon -= self.epsilon_decay


if __name__=="__main__":
    import gym
    env = gym.make("Hopper-v2")
    agent = DDPG(env)
    #agent.loadCheckpoint(Path to checkpoint)
    agent.train(animate=True)