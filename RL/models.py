import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, numStates, numActions, hidden1 = 128, hidden2 = 128, hidden3 = 64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(numStates, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, numActions)
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        out = self.tanh(out)
        return out
        
class Critic(nn.Module):
    def __init__(self, numStates, numActions, hidden1 = 128, hidden2 = 128, hidden3 = 64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(numStates+numActions, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        
        self.relu = nn.ReLU()
        
    def forward(self, states_actions):
        out = self.fc1(states_actions)
        out = self.relu(out)
        
        out = self.fc2(out)
        out = self.relu(out)
        
        out = self.fc3(out)
        out = self.relu(out)
        
        out = self.fc4(out)
        return out