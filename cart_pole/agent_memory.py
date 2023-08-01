import random
import torch
from collections import deque


class AgentMemory:
    def __init__(self, max_size) -> None:
        self.memory = deque([], maxlen=max_size)

    def update(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: torch.cat(x, 0), samples)
