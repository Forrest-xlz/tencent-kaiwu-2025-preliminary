import numpy as np
from agent_diy.conf.conf import Config


class MemoryMap:
    def __init__(self, logger=None):
        self.logger = logger
        self.memory_map = np.zeros(Config.MAP_HEIGHT * Config.MAP_WIDTH, dtype=np.float32)

    def reset(self):
        self.memory_map.fill(0)

    def update(self, x, z):
        self.memory_map[z * Config.MAP_WIDTH + x] += 0.2

        
