import numpy as np
import torch
import torch.optim as optim
from networks.CNN import CNN
from networks.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ValueNetImage(CNN):
  """Value estimation network."""
  def __init__(self, input_channels, hidden_dim, lr=1e-3):
    """Creates a new value estimation network.
    
        input_channels: number of input channels
        hidden_dim: number of hidden units
        action_dim: number of action dimensions
        lr: learning rate
    """
    super().__init__(input_channels, hidden_dim, 1)
    
    # Reinitialize optimizer because we added a parameter
    self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

    # Move to device
    self.to(device)