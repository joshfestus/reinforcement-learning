"""The game as an environment that a player, or an AI, can learn from.

An environment answers three questions: which actions are allowed, what
the player sees, and how good each frame was (the reward).
"""

import numpy as np

from brickgame.game import (
    BALL_SPEED,
    HEIGHT,
    PADDLE_WIDTH,
    WIDTH,
    BrickGame,
)
from brickgame.levels import COLUMNS, MAX_ROWS

BRICK_REWARD = 1.0
LOST_LIFE_REWARD = -1.0

# Losing a life is only punished once the ball has fallen, long after
# the move that caused it. This small reward arrives at the moment a
# return succeeds. It stays small so that bricks remain the real goal.
PADDLE_HIT_REWARD = 0.1

# A thousand wasted frames cost as much as one brick is worth, so a quick
# clear scores more than a slow one.
FRAME_COST = 0.001

# Where each number sits in the observation. The bricks follow these
# seven, one number per position in the grid, row by row.
PADDLE_X = 0
BALL_X = 1
BALL_Y = 2
BALL_DX = 3
BALL_DY = 4
BALL_FROM_PADDLE = 5
PADDLE_SIZE = 6
BRICKS_START = 7


class BrickEnv:
    def __init__(self):
        self.game = BrickGame()

    def observe(self):
        """Describe the game as numbers, most of them between -1 and 1.

        This list is everything the player knows about the game.
        """
        game = self.game
        paddle_center = game.paddle_x + game.paddle_width / 2
        numbers = [
            paddle_center / WIDTH * 2 - 1,
            game.ball_x / WIDTH * 2 - 1,
            game.ball_y / HEIGHT * 2 - 1,
            game.ball_dx / BALL_SPEED,
            game.ball_dy / BALL_SPEED,
            (game.ball_x - paddle_center) / WIDTH,
            game.paddle_width / PADDLE_WIDTH,
        ]

        bricks = np.zeros((MAX_ROWS, COLUMNS), dtype=np.float32)
        for brick in game.bricks:
            bricks[brick.row, brick.column] = 1.0

        numbers = np.array(numbers, dtype=np.float32)
        return np.concatenate([numbers, bricks.flatten()])

    def reset(self):
        """Start a new game and return what the player sees."""
        self.game.reset()
        return self.observe()

    def step(self, action):
        """Play one frame. Returns what the player sees now, the reward
        for this frame, and whether the game is over."""
        game = self.game
        score_before = game.score
        lives_before = game.lives
        hits_before = game.paddle_hits

        game.step(action)

        bricks = game.score - score_before
        lives_lost = lives_before - game.lives
        returns = game.paddle_hits - hits_before
        reward = (
            BRICK_REWARD * bricks
            + LOST_LIFE_REWARD * lives_lost
            + PADDLE_HIT_REWARD * returns
            - FRAME_COST
        )
        return self.observe(), reward, game.game_over
