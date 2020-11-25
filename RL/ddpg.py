import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable

import pickle

from memory import Memory
from random_process import OrnsteinUhlenbeckProcess

from models import (Actor, Critic)

from gym_pybullet_drones.utils.utils import *

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, numStates, numActions):
        
        self.lr_actor = 0.0001
        self.lr_critic = 0.001

        self.numStates = numStates
        self.numActions= numActions
        
        # Create Actor and Critic Network
        self.actor = Actor(self.numStates, self.numActions)
        self.actor_target = Actor(self.numStates, self.numActions)
        self.actor_optim  = Adam(self.actor.parameters(), lr=self.lr_actor)

        self.critic = Critic(self.numStates, self.numActions)
        self.critic_target = Critic(self.numStates, self.numActions)
        self.critic_optim  = Adam(self.critic.parameters(), lr=self.lr_critic)

        hard_update(self.actor_target, self.actor) # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)
        
        #Create replay buffer
        self.memory = Memory(max_size=1000000)
        self.random_process = OrnsteinUhlenbeckProcess(theta=0.15, size=self.numActions, mu=0.0, sigma=0.2)

        # Hyper-parameters
        self.batch_size = 32
        self.tau = 0.001
        self.discount = 0.99
        self.epsilon = 1
        self.min_epsilon = 0.5
        self.depsilon = 1/100000

        # 
        self.is_training = True

    def update_policy(self):
        # Sample batch
        #state_batch, action_batch, reward_batch, \
        #next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample(self.batch_size)
        

        # Prepare for the target q batch
        next_state_tensor = to_tensor(next_state_batch)
        next_state_actions = self.actor_target(next_state_tensor)
        with torch.no_grad():
            next_q_values = self.critic_target(torch.cat([next_state_tensor,next_state_actions],1))

        target_q_batch = to_tensor(reward_batch) + \
            self.discount*to_tensor(terminal_batch.astype(np.float))*next_q_values

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic(torch.cat([to_tensor(state_batch), to_tensor(action_batch)],1))
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        state_tensor = to_tensor(state_batch)
        state_actions = self.actor(state_tensor)
        policy_loss = -self.critic(torch.cat([state_tensor,state_actions],1))

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

    def select_action(self, state, decay_epsilon=True):
        action = to_numpy(self.actor(torch.from_numpy(np.array([state])).float())).squeeze(0)
        action += self.is_training*max(self.epsilon, self.min_epsilon)*self.random_process.sample()
        action = np.clip(action, -1., 1.)

        if decay_epsilon:
            self.epsilon -= self.depsilon

        return action
        
    def observe(self, s, a, r, ns, done):
        if self.is_training:
            self.memory.push(s, a, r, ns, done)

    def reset(self):
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}_actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}_critic.pkl'.format(output))
        )
        
    def load_experience(self, file_name):
        if file_name is None: return
        with open(file_name,'rb') as f:
            self.memory = pickle.load(f)

    def save_model(self, output):
        torch.save(
            self.actor.state_dict(),
            '{}_actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}_critic.pkl'.format(output)
        )
        with open('{}_experience.pkl'.format(output),'wb') as f:
            pickle.dump(self.memory,f)