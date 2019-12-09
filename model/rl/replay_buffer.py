import numpy as np
import random


class ReplayBuffer():
    def __init__(self, buffer_size=5000):
        self.buffer = []
        self.buffer_size = buffer_size
        self.item_dim = 0

    def add(self, experience):
        self.item_dim = experience.shape[-1]
        if len(self.buffer)+len(experience) >= self.buffer_size:
            self.buffer[0:(len(self.buffer)+len(experience))-self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, self.item_dim])

    def get_buffer(self):
        return np.reshape(np.array(self.buffer),[len(self.buffer), self.item_dim])

