import numpy as np
import math
import torch

class AsyncControl:
    def __init__(self, num_envs, num_agents):
        self.num_envs = num_envs
        self.num_agents = num_agents

        self.reset()
    
    def reset(self):
        # Count the number of steps taken by each agent
        self.cnt = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
        # Active status of each agent
        self.active = np.ones((self.num_envs, self.num_agents), dtype=np.int32)
        # Perivious step status of each agent
        self.prev_active = np.zeros((self.num_envs, self.num_agents), dtype=np.int32)
    
    def step(self, obs, actions):
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                # Mark the agent as inactive first
                self.active[e, a] = 0
                self.prev_active[e, a] = 0
                # Check if the agent has taken an action
                if a in actions[e].keys():
                    # If the agent has taken an action, mark it as active
                    self.prev_active[e, a] = 1
                # Check the activate agent
                if a in obs[e].keys():
                    # If the agent is active, mark it as active
                    self.active[e, a] = 1
                    self.cnt[e, a] += 1
    
    def active_agents(self):
        '''Return the active agents'''
        ret = []
        for e in range(self.num_envs):
            for a in range(self.num_agents):
                if self.active[e, a]:
                    ret.append((e, a, self.cnt[e, a]))
        return ret