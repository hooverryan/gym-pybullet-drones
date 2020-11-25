import random
import numpy as np
from collections import deque

class Memory:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        state_batch      = np.array(state_batch).reshape(batch_size,-1)
        action_batch     = np.array(action_batch).reshape(batch_size,-1)
        reward_batch     = np.array(reward_batch).reshape(batch_size,-1)
        next_state_batch = np.array(next_state_batch).reshape(batch_size,-1)
        done_batch       = np.array(done_batch).reshape(batch_size,-1)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)