import random
from collections import deque, namedtuple

Transition = namedtuple("Transition", ("state", "action", "reward", "new_state"))


class AgentMemory:
    def __init__(self, max_size) -> None:
        self.memory = deque([], maxlen=max_size)

    def update(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
