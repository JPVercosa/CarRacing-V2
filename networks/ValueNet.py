import numpy as np
import torch
import torch.optim as optim
from networks.CNN import CNN
from networks.utils import *
import os
from datetime import datetime

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

    def save(self, filepath):
        """Save the model parameters to a file."""
        prefix = datetime.now().strftime("%m-%d_%H-%M-%S")
        path = os.path.join(filepath, prefix + '-VNI_critic.pt')
        torch.save(self.state_dict(), path)

    def load(self, filepath, prefix):
        """Load the model parameters from a file."""
        path = os.path.join(filepath, prefix + '-VNI_critic.pt')
        self.load_state_dict(torch.load(path))