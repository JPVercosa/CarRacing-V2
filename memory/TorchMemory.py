import torch

class Memory:
    """Replay memory

       METHODS
           add    -- Add transition to memory.
           sample -- Sample minibatch from memory.
    """
    def __init__(self, input_channels, env_states, action_dims, size=1000000):
        """Creates a new replay memory.

           Memory(input_channels, action_dims) creates a new replay memory for storing
           transitions with `input_channels` observation dimensions and `action_dims`
           action dimensions. It can store 1000000 transitions.

           Memory(input_channels, action_dims, size) additionally specifies how many
           transitions can be stored.
        """

        self.s = torch.zeros(size, 1, env_states[0], env_states[1])
        self.a = torch.zeros(size, action_dims)
        self.r = torch.zeros(size, 1)
        #self.sp = torch.zeros(size, input_channels, env_states[0], env_states[1])
        self.terminal = torch.zeros(size, 1)
        self.v = torch.zeros(size, 1)
        self.logp = torch.zeros(size, 3)
        self.adv = torch.zeros(size, 1)
        self.rtg = torch.zeros(size, 1)
        self.i = 0
        self.n = 0
        self.size = size
        self.input_channels = input_channels
    
    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""
        return self.n
    
    def add(self, s, a, r, terminal, v=0, logp=0):
        """Adds a transition to the replay memory.

           Memory.add(s, a, r, sp, terminal) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `terminal`
           specifies whether the episode finished at terminal absorbing
           state `sp`.

           Memory.add(s, a, r, sp, terminal, v, logp) additionally records
           the value of state s and log-probability of taking action a.
        """
        
        self.s[self.i, :] = s.clone().detach()
        self.a[self.i, :] = a.clone().detach()
        self.r[self.i, :] = r.clone().detach()
        #self.sp[self.i, :] = self.s[(self.i - 1) % self.size, :]
        self.terminal[self.i, :] = terminal.clone().detach()
        self.logp[self.i, :] = torch.tensor(logp)
        
        self.i = (self.i + 1) % self.size
        if self.n < self.size:
            self.n += 1
    
    def sample(self, size):
        """Get random minibatch from memory.

        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        if self.n <= self.input_channels:
            raise ValueError(f"Memory has less than {self.input_channels} transitions.")
        
        assert len(self.s) == len(self.a) == len(self.r) == len(self.terminal), "Memory arrays must be of the same length."

        idx = torch.randint(0, self.n - self.input_channels - 1, (size,))
        
        s = torch.zeros((size, self.input_channels, *self.s.shape[2:]))
        sp = torch.zeros((size, self.input_channels, *self.s.shape[2:]))

        for i in range(size):
            for j in range(self.input_channels):
                s[i, j, :, :] = self.s[idx[i] + j, :, :]
                sp[i, j, :, :] = self.s[idx[i] + j + 1, :, :]

        a = self.a[idx + self.input_channels]
        r = self.r[idx + self.input_channels]
        terminal = self.terminal[idx + self.input_channels]
        
        assert s.shape == (size, self.input_channels, *self.s.shape[2:]), f"Expected shape {(size, self.input_channels, *self.s.shape[2:])}, but got {s.shape}"
        assert a.shape == (size, self.a.shape[1]), f"Expected shape {(size, self.a.shape[1])}, but got {a.shape}"
        assert r.shape == (size, 1), f"Expected shape {(size, 1)}, but got {r.shape}"
        assert sp.shape == (size, self.input_channels, *self.s.shape[2:]), f"Expected shape {(size, self.input_channels, *self.s.shape[2:])}, but got {sp.shape}"
        assert terminal.shape == (size, 1), f"Expected shape {(size, 1)}, but got {terminal.shape}"

        return s, a, r, sp, terminal
        
    def reset(self):
        """Reset memory."""

        self.i = 0
        self.n = 0

    def to(self, device):
        """Move memory to the specified device."""
        self.s = self.s.to(device)
        self.a = self.a.to(device)
        self.r = self.r.to(device)
        self.terminal = self.terminal.to(device)
        self.v = self.v.to(device)
        self.logp = self.logp.to(device)
        self.adv = self.adv.to(device)
        self.rtg = self.rtg.to(device)
