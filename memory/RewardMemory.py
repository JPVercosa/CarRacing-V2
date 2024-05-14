import numpy as np

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

        self.s = np.ndarray([size, input_channels, env_states[0], env_states[1]])
        self.a = np.ndarray([size, action_dims])
        self.r = np.ndarray([size, 1])
        self.sp = np.ndarray([size, input_channels, env_states[0], env_states[1]])
        self.terminal = np.ndarray([size, 1])
        self.v = np.ndarray([size, 1])
        self.logp = np.ndarray([size, 3])
        self.adv = np.ndarray([size, 1])
        self.rtg = np.ndarray([size, 1])
        self.i = 0
        self.n = 0
        self.size = size
    
    def __len__(self):
        """Returns the number of transitions currently stored in the memory."""

        return self.n
    
    def add(self, s, a, r, sp, terminal, v=0, logp=0):
        """Adds a transition to the replay memory.
        
           Memory.add(s, a, r, sp, terminal) adds a new transition to the
           replay memory starting in state `s`, taking action `a`,
           receiving reward `r` and ending up in state `sp`. `terminal`
           specifies whether the episode finished at terminal absorbing
           state `sp`.

           Memory.add(s, a, r, sp, terminal, v, logp) additionally records
           the value of state s and log-probability of taking action a.
        """

        self.s[self.i, :] = s
        self.a[self.i, :] = a
        self.r[self.i, :] = r
        self.sp[self.i, :] = sp
        self.terminal[self.i, :] = terminal
        self.v[self.i, :] = v
        self.logp[self.i, :] = logp
        
        self.i = (self.i + 1) % self.size
        if self.n < self.size:
            self.n += 1
    
    def sample(self, size):
        """Get random minibatch from memory.
        
        s, a, r, sp, done = Memory.sample(batch) samples a random
        minibatch of `size` transitions from the replay memory. All
        returned variables are vectors of length `size`.
        """

        idx = np.random.randint(0, self.n, size)

        return self.s[idx], self.a[idx], self.r[idx], self.sp[idx], self.terminal[idx]
        
    def reset(self):
        """Reset memory."""

        self.i = 0
        self.n = 0