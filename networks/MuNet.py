import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.functional import F
from networks.CNN import CNN
from networks.utils import *
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MuNetImage(CNN):
    """Deterministic policy network."""
    def __init__(self, input_channels, hidden_dim, action_dim):
        """Creates a new deterministic policy network.
        
           Mu(state_dims, action_dims) creates a network for states with
           `state_dims` dimensions and `action_dims` action dimensions.
           Mu(state_dims, action_dims, hiddens, lr) additionally specifies sizes of
           the hidden layers in `hiddens`, as well as the learning rate `lr`.
           
           The action is in the range [-1, 1].
        """
        super().__init__(input_channels, hidden_dim, action_dim)
        
    def update(self, s, q_net):
        """Train network.
        
           update(s, q_net) performs one gradient descent step on the network
           to increase the value of q_net(s, Mu(s)).
           
           `s` is a matrix with a batch of vectors (one vector per row).
           
           EXAMPLE
               >>> cq = CQ(3, 1)
               >>> mu = Mu(3, 1)
               >>> mu.update([[1, 2, 3], [4, 5, 6]], cq)
        """           
        loss = -q_net.forward(s, self.forward(s), cpu=False).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def soft_update(self, target_net, tau):
        for target_param, param in zip(target_net.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)