import numpy as np
from gym.spaces import Box, Discrete

from .game import PLAY_AREA_RESIZED
from .types import Action


class FakeGameEnv:
    def __init__(self):
        self.observation_space = Box(
            0, 255, (2, PLAY_AREA_RESIZED[0], PLAY_AREA_RESIZED[1]), dtype=np.uint8
        )
        self.action_space = Discrete(len(Action))
        self.metadata = None

    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=np.uint8)
