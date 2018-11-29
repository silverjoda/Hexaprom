"""
Replay Buffer 
Author: Sameera Lanka
Website: https://sameera-lanka.com
"""

import random
from collections import deque
import numpy as np
import torch as T
MINIBATCH_SIZE = 64

class Buffer:
    def __init__(self, buffer_size):
        self.limit = buffer_size
        self.data = deque(maxlen=self.limit)
        
    def __len__(self):
        return len(self.data)
    
    def sample_batch(self, batchSize):
        if len(self.data) < batchSize:
            raise ValueError('Not enough entries to sample without replacement.')
        else:
            batch = random.sample(self.data, batchSize)
            curState = [element[0] for element in batch]
            action = [element[1] for element in batch]
            nextState = [element[2] for element in batch]
            reward = [element[3] for element in batch]
            terminal = [element[4] for element in batch]

        curState = T.FloatTensor(np.array(curState).astype(np.float32))
        action = T.FloatTensor(np.array(action).astype(np.float32))
        nextState = T.FloatTensor(np.array(nextState).astype(np.float32))
        reward = T.FloatTensor(np.array(reward).astype(np.float32)).unsqueeze(1)

        return curState, action, nextState, reward, terminal
                  
    def append(self, element):
        self.data.append(element)  
