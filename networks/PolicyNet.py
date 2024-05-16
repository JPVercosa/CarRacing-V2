import numpy as np
import torch
import torch.optim as optim
from torch.distributions.normal import Normal
from networks.CNN import CNN
from networks.utils import *
import datetime, os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyNetImage(CNN):
    """Stochastic policy network."""
    def __init__(self, input_channels, hidden_dim, action_dims, lr=3e-4):
        """Creates a new stochastic policy network.
        
            input_channels: number of input channels
            hidden_dim: number of hidden units
            action_dims: number of action dimensions
            lr: learning rate
        """
        super().__init__(input_channels, hidden_dim, action_dims)
        log_std = -0.5 * np.ones(action_dims, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std).to(device))
        self.tanh = torch.nn.Tanh() 
        # Reinitialize optimizer because we added a parameter
        self.optimizer = optim.AdamW(self.parameters(), lr=lr, amsgrad=True)

        # Move to device
        self.to(device)

    def forward(self, s, a_prev=None, cpu=False):
        """Perform inference.
        
           Pi.forward(s) returns an action sampled from the policy distribution at
           state `s`, along with the log-probability of that action.
           Pi.forward(s, a_prev) returns an action sampled from the policy
           distribution at state `s`, along with the log-probability of taking
           action `a_prev` at that same state.
           
           `s` can be either a single vector, or a matrix with a batch of
           vectors. In the latter case, `a_prev`, when given, must also be a batch
           of vectors.
        
           EXAMPLE
               >>> pi = Pi(3, 1)
               >>> pi.forward([1, 2, 3])
               (array([0.3648948], dtype=float32), array([-0.48902923], dtype=float32))
               >>> pi.forward([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]])
               (array([[-0.19757865],
                       [ 0.336668  ]], dtype=float32),
                array([[-0.45469385],
                       [-0.56612885]], dtype=float32))
        """
        mu = self.tanh(super().forward(s))
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        a = dist.sample()
        if a_prev is None:
            # Return log probability of taking sampled action
            logp = dist.log_prob(a)
        else:
            # Return log probability of taking action a_prev
            logp = dist.log_prob(ttf(a_prev))
            
        if cpu:
            a = a.cpu().detach().numpy()
            logp = logp.cpu().detach().numpy()
        return a, logp

    def update(self, s, a_prev, logp_a, adv, clip_ratio=0.2):
        """Train network.
        
           update(s, a_prev, logp_a, adv) performs one gradient descent step
           on the network to minimize the clipped PPO objective function,
           using the advantages `adv` of taking actions `a_prev` at
           states `s`, with probabilities `a_prev`.
           update(s, a_prev, logp_a, adv, clip_ratio) additionally specifies
           the `clip_ratio` that constrains the size of the update.
           
           `s`, `a_prev`, `logp_a` and `adv` are matrices with batches of
           vectors (one vector per row).
           
           EXAMPLE
               >>> pi = Pi(3, 1)
               >>> pi.update([[1, 2, 3], [4, 5, 6]], [[0.3], [0.8]], [[-0.45], [-0.56]], [[1.12], [-0.23]])
        """           
        _, logp = self.forward(s, a_prev, cpu=False)
        ratio = torch.exp(logp - ttf(logp_a))
        adv_t = ttf(adv)
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * adv_t
        loss = -(torch.min(ratio * adv_t, clip_adv)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath):
        """Save the model parameters to a file."""
        prefix = datetime.now().strftime("%m-%d_%H-%M-%S")
        path = os.path.join(filepath, prefix + '-VNI.pt')
        torch.save(self.state_dict(), path)

    def load(self, filepath, prefix):
        """Load the model parameters from a file."""
        path = os.path.join(filepath, prefix + '-VNI.pt')
        self.load_state_dict(torch.load(path))
    