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

import os

# Files
from HER.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from HER.replaybuffer_her import Buffer
from HER.actorcritic_her import Actor, Critic

torch.set_num_threads(2)

# Hyperparameters
ACTOR_LR = 0.0005
CRITIC_LR = 0.001
MINIBATCH_SIZE = 64
NUM_EPISODES = 100000
MU = 0
SIGMA = 0.2
BUFFER_SIZE = 10000000
DISCOUNT = 0.95
TAU = 0.001
WARMUP = 320
EPSILON = 1.0
EPSILON_DECAY = 1e-5


class DDPG:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.obs_dim
        self.act_dim = env.act_dim
        self.actor = Actor(self.env)
        self.critic = Critic(self.env)
        self.targetActor = deepcopy(self.actor)
        self.targetCritic = deepcopy(self.critic)
        self.actorOptim = optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.criticOptim = optim.Adam(self.critic.parameters(), lr=CRITIC_LR)
        self.criticLoss = nn.MSELoss()
        self.noise = OUNoise(mu=np.zeros(self.act_dim), sigma=SIGMA)
        self.replayBuffer = Buffer(BUFFER_SIZE)
        self.batchSize = MINIBATCH_SIZE
        self.gamma = DISCOUNT
        self.warmup = WARMUP
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.rewardgraph = []


    def getQTarget(self, nextStateBatch, rewardBatch, terminalBatch):
        Qvals = self.targetCritic(nextStateBatch, self.targetActor(nextStateBatch))

        y_i = []
        for r,t,q in zip(rewardBatch, terminalBatch, Qvals):
            if t:
                y_i.append(r.unsqueeze(0))
            else:
                y_i.append(r + self.gamma * q)

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
        noise = (self.epsilon * T.randn(self.act_dim))

        action = self.actor(s)
        actionNoise = action + noise
        return actionNoise.detach()


    def train(self, animate=False):

        print('Starting training...')
        
        for i in range(NUM_EPISODES):
            obs = self.env.reset()
            goal = self.env.goal
            done = False
            ep_reward = 0

            observations = []
            actions = []
            new_observations = []
            terminals = []

            while not done:

                # Get action
                c_obs = T.FloatTensor(np.concatenate([obs, goal]).astype(np.float32)).unsqueeze(0)

                action = self.getMaxAction(c_obs)

                # Step episode
                obs_new, r, done, _ = self.env.step(action.numpy())

                if animate:
                    self.env.render()

                # Add new data
                observations.append(obs)
                actions.append(action)
                terminals.append(done)
                new_observations.append(obs_new)

                # Add standard experience replay jobby
                next_c_obs = T.FloatTensor(np.concatenate([obs_new, goal]).astype(np.float32)).unsqueeze(0)

                # Add transition
                self.replayBuffer.append((c_obs, action, next_c_obs, r, done))

                ep_reward += r
                obs = obs_new

                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)

                    curStateBatch = T.cat(curStateBatch)
                    actionBatch = T.cat(actionBatch)
                    nextStateBatch = T.cat(nextStateBatch)
                    rewardBatch = T.from_numpy(np.asarray(rewardBatch).astype(np.float32))

                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)

                    # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = (qPredBatch - qTargetBatch).pow(2).mean()
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
                        #print(qPredBatch, qTargetBatch)
                        print("Episode {}/{}, ep reward: {}, actr loss: {}, critic loss: {}, success rate: {}".format(i,
                                                                                                                      NUM_EPISODES,
                                                                                                                      ep_reward,
                                                                                                                      actorLoss,
                                                                                                                      criticLoss,
                                                                                                                      env.success_rate))

            final_pose = env.get_pose(obs)
            self.epsilon -= self.epsilon_decay

            # Append all hindsight transitions
            for o,a,t,o_ in zip(observations,actions,terminals,new_observations):
                c_obs = T.FloatTensor(np.concatenate([o, final_pose]).astype(np.float32)).unsqueeze(0)
                next_c_obs = T.FloatTensor(np.concatenate([o_, final_pose]).astype(np.float32)).unsqueeze(0)

                # Reward is 1 if we reached our terminal goal (in HER we always get rew 1 when we reach terminal goal)
                r = 1. if t else -0.01

                # Add transition
                self.replayBuffer.append((c_obs, a, next_c_obs, r, t))





if __name__=="__main__":
    # TODO: FIX BY ADDING PHASE 1 WHERE WE ADD NORMAL EXPERIENCE REPLAY AND 2ND PHASE WHERE WE ADD HINDSIGHT TRANSITIONS
    import gym
    env = AntG()
    agent = DDPG(env)
    #agent.loadCheckpoint(Path to checkpoint)
    agent.train(animate=True)