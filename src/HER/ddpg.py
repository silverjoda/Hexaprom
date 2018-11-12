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


# Lib
import numpy as np
import random
from copy import deepcopy

import matplotlib.pyplot as plt

from IPython import display
import os

# Files
from HER.noise import OrnsteinUhlenbeckActionNoise as OUNoise
from HER.replaybuffer import Buffer
from HER.actorcritic import Actor, Critic

# Hyperparameters
ACTOR_LR = 0.0001
CRITIC_LR = 0.001
MINIBATCH_SIZE = 64
NUM_EPISODES = 10000
MU = 0
SIGMA = 0.2
BUFFER_SIZE = 1000000
DISCOUNT = 0.9
TAU = 0.001
WARMUP = 70
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
        """Inputs: Batch of next states, rewards and terminal flags of size self.batchSize
            Calculates the target Q-value from reward and bootstraped Q-value of next state
            using the target actor and target critic
           Outputs: Batch of Q-value targets"""
        
        targetBatch = torch.FloatTensor(rewardBatch).cuda() 
        nonFinalMask = torch.ByteTensor(tuple(map(lambda s: s != True, terminalBatch)))
        nextStateBatch = torch.cat(nextStateBatch)
        nextActionBatch = self.targetActor(nextStateBatch)
        nextActionBatch.volatile = True
        qNext = self.targetCritic(nextStateBatch, nextActionBatch)  
        
        nonFinalMask = self.discount * nonFinalMask.type(torch.cuda.FloatTensor)
        targetBatch += nonFinalMask * qNext.squeeze().data
        
        return Variable(targetBatch, volatile=False)

    
    def updateTargets(self, target, original):
        """Weighted average update of the target network and original network
            Inputs: target actor(critic) and original actor(critic)"""
        
        for targetParam, orgParam in zip(target.parameters(), original.parameters()):
            targetParam.data.copy_((1 - TAU)*targetParam.data + \
                                          TAU*orgParam.data)

            
  
    def getMaxAction(self, s):
        """Inputs: Current state of the episode
            Returns the action which maximizes the Q-value of the current state-action pair"""
       
        noise = self.epsilon * T.from_numpy(self.noise())
        action = self.actor(s)
        actionNoise = action + noise
        return actionNoise

    def sampleGoal(self):
        pass

    def train(self):

        print('Starting training...')
        
        for i in range(NUM_EPISODES):
            obs = self.env.reset()
            done = False
            ep_reward = 0

            # Sample goal
            goal = self.sampleGoal()
            
            while not done:

                # Get action
                c_obs = T.cat([obs, goal], 1)
                c_obs = T.from_numpy(c_obs).unsqueeze(0)
                action = self.actor.forward(c_obs)
                
                # Step episode
                new_obs, r, done, _ = self.env.step(action.data)
                new_c_obs = T.cat([new_obs, goal], 1)

                ep_reward += r
                
                # Update replay bufer
                self.replayBuffer.append((c_obs, action, new_c_obs, r, done))
                
                # Training loop
                if len(self.replayBuffer) >= self.warmup:
                    
                    curStateBatch, actionBatch, nextStateBatch, \
                    rewardBatch, terminalBatch = self.replayBuffer.sample_batch(self.batchSize)
                    curStateBatch = T.cat(curStateBatch)
                    actionBatch = T.cat(actionBatch)
                    
                    qPredBatch = self.critic(curStateBatch, actionBatch)
                    qTargetBatch = self.getQTarget(nextStateBatch, rewardBatch, terminalBatch)
                    
                    # Critic update
                    self.criticOptim.zero_grad()
                    criticLoss = self.criticLoss(qPredBatch, qTargetBatch)
                    print('Critic Loss: {}'.format(criticLoss))
                    criticLoss.backward()
                    self.criticOptim.step()
            
                    # Actor update
                    self.actorOptim.zero_grad()
                    actorLoss = -T.mean(self.critic(curStateBatch, self.actor(curStateBatch)))
                    print('Actor Loss: {}'. format(actorLoss))
                    actorLoss.backward()
                    self.actorOptim.step()
                    
                    # Update Targets
                    self.updateTargets(self.targetActor, self.actor)
                    self.updateTargets(self.targetCritic, self.critic)
                    self.epsilon -= self.epsilon_decay

                obs = new_obs
                    
            if i % 20 == 0:
                print("Episode {}/{}, episode reward: {}".format(i, NUM_EPISODES, ep_reward))


if __name__=="__main__":
    import gym
    env = gym.make("FetchReacher-v1")
    agent = DDPG(env)
    #agent.loadCheckpoint(Path to checkpoint)
    agent.train()