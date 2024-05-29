import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import torch.nn.functional as F
from networks.MLP import MLP
from networks.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
  def __init__(self, input_channels, hidden_dim, action_dim=1, lr=1e-3):
    """
    Creates a new convolutional neural network.

    CNN(input_channels, hidden_dim, action_dim) creates a CNN with `input_channels`
    """
    super(CNN, self).__init__()
  
    
    # Convolutional layers
    self.cnn_base = nn.Sequential(
      nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
      nn.ReLU(),
      nn.Conv2d(8, 16, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(16, 32, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(32, 64, kernel_size=3, stride=2),
      nn.ReLU(),
      nn.Conv2d(64, 128, kernel_size=3, stride=1),
      nn.ReLU(),
      nn.Conv2d(128, 256, kernel_size=3, stride=1),
      nn.ReLU()
    )

    # Additional linear layer to reduce the concatenated feature size
    self.reduce_dim = nn.Linear(259, 256)
    self.reduce_dim2 = nn.Linear(257, 256)

    # Fully connected layers
    self.fc = nn.Sequential(
      nn.Linear(256, hidden_dim),
      nn.ReLU(),
      nn.Linear(hidden_dim, action_dim)
    )

    self.criterion = nn.MSELoss()
    self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)
    # self.apply(self._weights_init)

  def _weights_init(self, m):
    if isinstance(m, nn.Conv2d):
      nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
      nn.init.constant_(m.bias, 0.1)

  def forward(self, x, a=None):
    # print(f"1-X is instance of tensor: {isinstance(x, torch.Tensor)}")
    # print(f"1-A is instance of tensor: {isinstance(a, torch.Tensor)}")
    if isinstance(x, torch.Tensor):
      x.requires_grad_()
    if isinstance(a, torch.Tensor):
      a.requires_grad_()
    #print("Input shape: ", x.shape)
    output = self.cnn_base(ttf(x))
    # print(f"2-Output is instance of tensor: {isinstance(output, torch.Tensor)}")
    #print("After CNN: ", output.shape)
    output = output.view(output.size(0), -1)
    #print("After change view: ", output.shape)
    if a is not None:
      output = torch.cat((output, ttf(a)), dim=1)
      assert output.shape[1] == 257 or output.shape[1] == 259, f"Output shape: {output.shape}"
      if output.shape[1] == 257:
        output = self.reduce_dim2(output)
      elif output.shape[1] == 259:
        output = self.reduce_dim(output)
    #print("After concat: ", output.shape)

    output = self.fc(output)
    #print("After FC: ", output.shape)
    return output
  
  def update(self, inputs, targets, actions=None, cpu=True, retain_graph=False):
    """Train network.
    
    update(inputs, targets) performs one gradient descent step
    """

    # if not isinstance(inputs, torch.Tensor):
    #     inputs = torch.tensor(inputs, dtype=torch.float32).to(device) 
    # if not isinstance(targets, torch.Tensor):
    #     targets = torch.tensor(targets, dtype=torch.float32).to(device)
    # if actions is not None and not isinstance(actions, torch.Tensor):
    #     actions = torch.tensor(actions, dtype=torch.float32).to(device)
    
    # if inputs.requires_grad is False:
    #     inputs.requires_grad_()
        
    if actions is not None:
      outputs = self.forward(inputs, actions, cpu=cpu)
    else:
      outputs = self.forward(inputs)
    loss = self.criterion(outputs, ttf(targets))
    self.optimizer.zero_grad()
    loss.backward(retain_graph=retain_graph)
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
      
    
    
