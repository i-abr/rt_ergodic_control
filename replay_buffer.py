import random
import numpy as np
from itertools import compress


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def reset(self):
        self.buffer = []
        self.position = 0

    def push(self, state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity


    def sample(self, batch_size):
        if batch_size == -1:
            states = self.buffer[:batch_size]
        else:
            states = random.sample(self.buffer, batch_size)
        # batch = random.sample(self.buffer, batch_size)
        # states, = map(np.stack, zip(*batch))
        return states

    def __len__(self):
        return len(self.buffer)
