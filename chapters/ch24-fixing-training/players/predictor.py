"""A hand-written player that works out where the ball will come down.

While the ball rises it chases it, like the chaser. Once the ball is
falling it follows the ball's path ahead, bouncing it off the side walls,
and sends the paddle to the point where it will land. It sees the same
numbers an AI would see, and never learns.

It ignores bricks the ball might hit on the way down, so now and then it
guesses wrong.
"""

from brickgame.env import BALL_DX, BALL_DY, BALL_X, BALL_Y, PADDLE_X
from brickgame.game import (
    BALL_RADIUS,
    BALL_SPEED,
    HEIGHT,
    LEFT,
    PADDLE_Y,
    RIGHT,
    STAY,
    WIDTH,
)
from players.chaser import Chaser

SLACK = 4


def landing_point(x, y, dx, dy):
    """Where a falling ball will reach the paddle's height."""
    left_wall = BALL_RADIUS
    right_wall = WIDTH - BALL_RADIUS
    while y < PADDLE_Y - BALL_RADIUS:
        x += dx
        y += dy
        if x < left_wall:
            x = left_wall + (left_wall - x)
            dx = -dx
        if x > right_wall:
            x = right_wall - (x - right_wall)
            dx = -dx
    return x


class Predictor:
    name = "predictor"

    def __init__(self):
        self.chaser = Chaser()

    def act(self, observation):
        ball_dy = observation[BALL_DY] * BALL_SPEED
        if ball_dy < 0:
            return self.chaser.act(observation)

        # Turn the scaled numbers back into pixels.
        ball_x = (observation[BALL_X] + 1) / 2 * WIDTH
        ball_y = (observation[BALL_Y] + 1) / 2 * HEIGHT
        ball_dx = observation[BALL_DX] * BALL_SPEED
        paddle_x = (observation[PADDLE_X] + 1) / 2 * WIDTH

        target = landing_point(ball_x, ball_y, ball_dx, ball_dy)
        if target < paddle_x - SLACK:
            return LEFT
        if target > paddle_x + SLACK:
            return RIGHT
        return STAY
