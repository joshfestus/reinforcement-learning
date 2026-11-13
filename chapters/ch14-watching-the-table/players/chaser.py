"""A hand-written player that moves the paddle toward the ball.

It sees the same numbers an AI would see, and never learns.
"""

from brickgame.env import BALL_FROM_PADDLE
from brickgame.game import LEFT, RIGHT, STAY, WIDTH

# Without a few pixels of slack the paddle would twitch left and right
# when the ball is directly above it.
SLACK = 4 / WIDTH


class Chaser:
    name = "chaser"

    def act(self, observation):
        ball_from_paddle = observation[BALL_FROM_PADDLE]
        if ball_from_paddle < -SLACK:
            return LEFT
        if ball_from_paddle > SLACK:
            return RIGHT
        return STAY
