"""A player steered by a trained landing network.

It plays like the predictor, except that it does not work out the
ball's path itself. While the ball rises it chases it; while it falls
it asks the network where the ball will land and goes there.
"""

import numpy as np

from brickgame.env import BALL_DX, BALL_DY, BALL_X, BALL_Y, PADDLE_X
from brickgame.game import LEFT, RIGHT, STAY, WIDTH
from players.chaser import Chaser

SLACK = 4 / WIDTH * 2


class LandingPlayer:
    name = "landing network"

    def __init__(self, network):
        self.network = network
        self.chaser = Chaser()

    def act(self, observation):
        if observation[BALL_DY] < 0:
            return self.chaser.act(observation)

        ball = observation[[BALL_X, BALL_Y, BALL_DX, BALL_DY]]
        target = self.network.predict(np.array([ball]))[0]
        paddle = observation[PADDLE_X]
        if target < paddle - SLACK:
            return LEFT
        if target > paddle + SLACK:
            return RIGHT
        return STAY
