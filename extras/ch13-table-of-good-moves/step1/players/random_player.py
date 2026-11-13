"""A player that picks a random action every frame. It never learns."""

import random

from brickgame.game import ACTIONS


class RandomPlayer:
    name = "random"

    def __init__(self, seed=None):
        self.random = random.Random(seed)

    def act(self, observation):
        return self.random.choice(ACTIONS)
