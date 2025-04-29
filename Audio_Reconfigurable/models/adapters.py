import numpy as np
from tqdm import tqdm, trange
import torch
import torch.nn as nn

# Adapter is a two layer MLP with input layer, hidden layer, and output layer (256)
class Adapter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Adapter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        torch.nn.init.xavier_uniform_(self.fc1.weight) 
        self.activation = nn.LeakyReLU()
        self.fc2 = nn.Linear(hidden_dim, 256)
        torch.nn.init.xavier_uniform_(self.fc2.weight) 
    def forward(self, embedding):
        return self.fc2(self.activation(self.fc1(embedding)))