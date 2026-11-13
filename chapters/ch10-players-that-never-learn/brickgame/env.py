"""The game as an environment that a player, or an AI, can learn from.

An environment answers three questions: which actions are allowed, what
the player sees, and how good each frame was (the reward). It follows the
Gymnasium standard, so learning libraries that know Gymnasium can train
on it without changes.
"""

import gymnasium as gym
import numpy as np

from brickgame.game import (
    ACTIONS,
    BALL_SPEED,
    HEIGHT,
    PADDLE_WIDTH,
    START_LIVES,
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

# No brick broken for this long means the ball is bouncing in a loop.
STUCK_FRAMES = 10_000

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


class BrickEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, render_mode=None):
        self.game = BrickGame()
        self.render_mode = render_mode
        self.screen = None

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        size = BRICKS_START + COLUMNS * MAX_ROWS
        self.observation_space = gym.spaces.Box(
            low=-3.0, high=3.0, shape=(size,), dtype=np.float32
        )

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

    def information(self):
        return {
            "score": self.game.score,
            "level": self.game.level,
            "lives": self.game.lives,
            "frames": self.game.frames,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.game.random.seed(seed)
        self.game.reset()
        self.last_brick_frame = 0
        return self.observe(), self.information()

    def step(self, action):
        game = self.game
        score_before = game.score
        lives_before = game.lives
        hits_before = game.paddle_hits

        game.step(int(action))

        bricks = game.score - score_before
        lives_lost = lives_before - game.lives
        returns = game.paddle_hits - hits_before
        reward = (
            BRICK_REWARD * bricks
            + LOST_LIFE_REWARD * lives_lost
            + PADDLE_HIT_REWARD * returns
            - FRAME_COST
        )

        if bricks > 0:
            self.last_brick_frame = game.frames
        stuck = game.frames - self.last_brick_frame >= STUCK_FRAMES

        terminated = game.game_over
        truncated = not terminated and stuck

        info = self.information()
        if terminated:
            info["ending"] = "lost"
        elif stuck:
            info["ending"] = "stuck"
        info["lives_lost"] = START_LIVES - game.lives

        if self.render_mode == "human":
            self.render()
        return self.observe(), reward, terminated, truncated, info

    def render(self):
        if self.screen is None:
            from brickgame.render import Screen

            self.screen = Screen(title="Brick Breaker: AI playing")
        self.screen.draw(self.game)
        self.screen.wait(self.metadata["render_fps"])

    def close(self):
        if self.screen is not None:
            self.screen.close()
            self.screen = None
