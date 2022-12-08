import numpy as np
import random
import torch as T

class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, device):
        self.mem_size = max_size
        self.mem_cntr = 0
        # self.state_memory = np.zeros((self.mem_size, *input_shape),             #device = cuda
        #                              dtype=np.float32)
        # self.new_state_memory = np.zeros((self.mem_size, *input_shape),
        #                                  dtype=np.float32)
        #
        # self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        # self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        # self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)
        self.device = device

        self.state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.float32, device=self.device)         #device = cuda
        self.new_state_memory = T.zeros((self.mem_size, *input_shape), dtype=T.float32, device=self.device)
        self.action_memory = T.zeros(self.mem_size, dtype=T.int64, device=self.device)
        self.reward_memory = T.zeros(self.mem_size, dtype=T.float32, device=self.device)
        self.terminal_memory = T.zeros(self.mem_size, dtype=T.bool, device=self.device)

        # self.device = device

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        # batch = np.random.choice(max_mem, batch_size, replace=False)        #torch
        batch = random.sample(range(max_mem), batch_size)
        batch = T.tensor(batch)
        states = self.state_memory[batch] #.to(self.device)
        actions = self.action_memory[batch] #.to(self.device)
        rewards = self.reward_memory[batch] #.to(self.device)
        states_ = self.new_state_memory[batch] #.to(self.device)
        terminal = self.terminal_memory[batch] #.to(self.device)

        return states, actions, rewards, states_, terminal
