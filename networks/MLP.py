import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networks.utils import *


class MLP(nn.Module):
    """Basic multi-layer perceptron.
    
       METHODS
           forward     -- Perform inference.
           update      -- Train network.
           copyfrom    -- Copy weights.
    """
    def __init__(self, sizes, activation=nn.ReLU, output_activation=nn.Identity, lr=1e-3):
        """Creates a new multi-layer perceptron network.
        
           MLP(sizes) creates an MLP with len(sizes) layers, with the given `sizes`.
           MLP(sizes, activation, output_activation, lr) additionally specifies the
           activation function, output activation function (`torch.nn.*`) and learning
           rate.
        """
        super().__init__()
        self.sizes = sizes
        layers = []
        for j in range(len(sizes)-1):
            act = activation if j < len(sizes)-2 else output_activation
            layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
        self.net = nn.Sequential(*layers).to(device)
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
        self.criterion = nn.MSELoss()
        
    def forward(self, inputs, cpu=True):
        """Perform inference.
        
           MLP.forward(inputs) returns the output of the network for the given
           `inputs`.
        
           `inputs` can be either a single vector, or a matrix with a batch of
           vectors (one vector per row).
           
           EXAMPLE
               >>> mlp = MLP([3, 64, 64, 1])
               >>> mlp.forward([1, 2, 3])
               array([0.44394666], dtype=float32)
               >>> mlp.forward([[1, 2, 3], [4, 5, 6]])
               array([[0.44394666],
                      [0.9606236 ]], dtype=float32)
        """
        outputs = self.net(ttf(inputs))

        if cpu:
            outputs = outputs.cpu().detach().numpy()
        return outputs

    def update(self, inputs, targets):
        """Train network.
        
           update(inputs, targets) performs one gradient descent step
           on the network to approximate the mapping
           `inputs` -> `targets`.
           
           `inputs` and `targets` are matrices with batches of vectors
           (one vector per row).
           
           EXAMPLE
               >>> mlp = MLP([3, 64, 64, 1])
               >>> mlp.update([[1, 2, 3], [4, 5, 6]], [[1], [2]])
        """           
        outputs = MLP.forward(self, inputs, cpu=False)
        loss = self.criterion(outputs, ttf(targets))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
    def copyfrom(self, other):
        """Copies weights from a different instance of the same network.
        
           EXAMPLE
               >>> mlp1 = MLP([3, 64, 64, 1])
               >>> mlp2 = MLP([3, 64, 64, 1])
               >>> mlp1.copyfrom(mlp2)
        """  
        state_dict = self.state_dict()
        other_state_dict = other.state_dict()
        for key in state_dict:
            state_dict[key] = other_state_dict[key]
        self.load_state_dict(state_dict)