import time
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import torch

class Environment():
    """OpenAI Gym Environment wrapper.

       METHODS
           reset   -- Reset environment.
           step    -- Step environment.
           render  -- Visualize environment.
           close   -- Close visualization.
           
       MEMBERS
           states  -- Number of state dimensions.
           actions -- Actions, or number of action dimensions.
    """

    def reset(self):
        """Reset environment to start state.
        
           obs = env.reset() returns the start state observation.
        """
        return self.env.reset()[0]
    
    def step(self, u):
        """Step environment.
        
           obs, r, terminal, truncated, info = env.step(u) takes
           action u and returns the next state observation, reward,
           whether the episode terminated or was truncated, and
           extra information.
        """
        observation, reward, terminal, truncated, info = self.env.step(u)

        return (observation, reward, terminal, truncated, info)
    
    def render(self):
        """Render environment.
        
           env.render() renders the current state of the
           environment in a separate window.
           
           NOTE
               You must call env.close() to close the window,
               before creating a new environment; otherwise
               the kernel may hang.
        """
        return self.env.render()
    
    def close(self):
        """Closes the rendering window."""
        return self.env.close()    
    
class CarRacing(Environment):
    """FlappyBird-v0 environment."""
    def __init__(self, render_mode='human', img_stack=3):
        """Creates a new FlappyBird environment.
        
           EXAMPLE
               >>> env = FlappyBird()
               >>> print(env.states)
               3
               >>> print(env.actions)
               [0, 1]
        """
        if render_mode == 'human':
          self.env = gym.make('CarRacing-v2', render_mode=render_mode)
        else:
          self.env = gym.make('CarRacing-v2')
        
        #print(self.env.action_space)
        #print(type(self.env.observation_space.sample()))
        #print(self.env.observation_space.shape)
        self.action_space = self.env.action_space # Box([-1.  0.  0.], 1.0, (3,), float32)
        self.stack_list = []
        self.img_stack = img_stack
        self.states = self.env.observation_space.shape
        self.initial_buffer = 30
        self.out_of_track = 200
            

    def reset(self):
        self.stack_list = []
        self.initial_buffer = 30
        self.out_of_track = 100
        for _ in range(self.img_stack):
            img_rgb = Environment.reset(self)
            img_gray = self.rgb2gray(img_rgb)
            self.stack_list.append(img_gray)

        assert len(self.stack_list) == self.img_stack, f"Stack length is not {self.img_stack}, but {len(self.stack_list)}"

        return np.stack(self.stack_list, axis=1)

        

    def step(self, u, cpu=False):
        
        assert len(self.stack_list) == self.img_stack, f"Stack length is not {self.img_stack}, but {len(self.stack_list)}"
       
        #print(u)
        if cpu:
            u = u.cpu().detach().numpy()

        u = np.array(u)
        #print(u)
        img_rgb, r, terminal, truncated, info = Environment.step(self, u)
        #print(img_rgb, r, terminal, truncated, info)

        if self.initial_buffer < 0:
            if np.any(img_rgb[64:66, 43:53, 1] > 110):
                # Count the number of frames that have the green value greater than 110
                # print("Detecting green")
                pixels_off = img_rgb[64:66, 43:53, 1] > 110
                pixels_off = np.sum(pixels_off)
                if pixels_off > 12:
                    r -= pixels_off
                    self.out_of_track -= 1
                    #print(f"Pixels off: {pixels_off}")
                # if pixels_off <= 8:
                #     print(f"Pixels in: {pixels_off}")
                #     r += pixels_off
                #     self.out_of_track += 1
        self.initial_buffer -= 1
            
        img_gray = self.rgb2gray(img_rgb)
        self.stack_list.pop(0)
        self.stack_list.append(img_gray)

        if self.out_of_track <= 0:
            terminal = True
            r -= 5
            #print("Out of track")

        return np.stack(self.stack_list, axis=1), r, terminal, truncated, info
            

    def plotnetwork(self, network):
        """Plot network.

           plot(dq) plots the value function and induced policy of DQ network `dq`
           at bird velocity 0.
        """
        if network.sizes[0] != 3 or network.sizes[-1] != 2:
            raise ValueError("Network is not compatible with FlappyBird environment")

        xx, yy = np.meshgrid(np.linspace(0, 2, 64), np.linspace(-0.5, 0.5, 64))
        vv = np.zeros((64, 64))
        obs = np.hstack((np.reshape(xx , (xx.size, 1)),
                         np.reshape(yy , (yy.size, 1)),
                         np.reshape(vv , (vv.size, 1))
                       ))

        aval = [0, 1] 

        qq = network(obs)
        vf = np.reshape(np.amax(qq, axis=1), xx.shape)
        pl = np.vectorize(lambda x: aval[x])(np.reshape(np.argmax(qq, axis=1), xx.shape))

        fig, axs = plt.subplots(1,2)
        fig.subplots_adjust(right=1)

        h = axs[0].contourf(xx, yy, vf, 256)
        fig.colorbar(h, ax=axs[0])
        h = axs[1].contourf(xx, yy, pl, 256)
        fig.colorbar(h, ax=axs[1])

        axs[0].set_title('Value function')
        axs[1].set_title('Policy')

    @staticmethod
    def rgb2gray(rgb, norm=True):
      gray = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
      if norm:
        # normalize
        gray = gray / 128. - 1.

      # gray shape is (96, 96)
      # change to (1, 96, 96)
      gray = np.expand_dims(gray, axis=0)
      return gray