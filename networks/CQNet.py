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

class CQNetImage(CNN):
    """State-action value network with continuous actions."""
    def __init__(self, input_channels, hidden_dim, action_dim=1, lr=1e-3):
        """Creates a new state-action value network with continuous actions.
        
           CQ(state_dims, action_dims) creates a network for states with
           `state_dims` dimensions and `action_dims` action dimensions.
           CQ(state_dims, action_dims, hiddens, lr) additionally specifies sizes of
           the hidden layers in `hiddens`, as well as the learning rate `lr`.
        """
        super().__init__(input_channels, hidden_dim, action_dim)

    def forward(self, s, a, cpu=True):
        """Perform inference.
        
           CQ.forward(s, a) returns the value of action `a` at state `s`.
           
           `s` and `a` can be either single vectors, or matrices with batches of
           vectors.
        
           EXAMPLE
               >>> cq = CQ(3, 1)
               >>> cq.forward([1, 2, 3], [0.5])
               array([0.42496216], dtype=float32)
               >>> cq.forward([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]])
               array([[0.42496222],
                      [0.8978188 ]], dtype=float32)
        """
        #print(s.shape, a.shape)
        # a_expanded = np.expand_dims(np.expand_dims(a, axis=-1), axis=-1)
        # a_broadcasted = np.broadcast_to(a_expanded, s.shape)

        q = super().forward(ttf(s), ttf(a))
       
        if cpu:
            q = q.cpu().detach().numpy()
        return q
        
    def update(self, s, a, targets, cpu=True, retain_graph=False):
        """Train network.
        
           update(s, a, targets) performs one gradient descent step
           on the network to approximate the mapping
           (`s`, `a`) -> `targets`.
           
           `s`, `a`, and `targets` are matrices with batches of vectors
           (one vector per row).
           
           EXAMPLE
               >>> cq = DQ(3, 1)
               >>> cq.update([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]], [[0.5], [1.5]])
        """           
        super().update(s, targets, a, cpu=cpu, retain_graph=retain_graph)

    def soft_update(self, target_net, tau):
        for target_param, param in zip(target_net.parameters(), self.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)