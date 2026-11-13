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

    def __init__(
        self,
        render_mode=None,
        max_frames=None,
        paddle_hit_reward=PADDLE_HIT_REWARD,
        frame_cost=FRAME_COST,
        scaled=True,
        see_bricks=True,
        see_paddle_size=True,
    ):
        """The last three settings take parts of the observation away,
        to measure what each part is worth. A part that is not seen is
        always 0, so the observation keeps its 55 numbers."""
        self.game = BrickGame()
        self.render_mode = render_mode
        self.max_frames = max_frames
        self.paddle_hit_reward = paddle_hit_reward
        self.frame_cost = frame_cost
        self.scaled = scaled
        self.see_bricks = see_bricks
        self.see_paddle_size = see_paddle_size
        self.screen = None

        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        size = BRICKS_START + COLUMNS * MAX_ROWS
        limit = 3.0 if scaled else float(WIDTH)
        self.observation_space = gym.spaces.Box(
            low=-limit, high=limit, shape=(size,), dtype=np.float32
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
        if not self.scaled:
            # The same seven numbers in pixels, as the game keeps them.
            numbers = [
                paddle_center,
                game.ball_x,
                game.ball_y,
                game.ball_dx,
                game.ball_dy,
                game.ball_x - paddle_center,
                game.paddle_width,
            ]
        if not self.see_paddle_size:
            numbers[PADDLE_SIZE] = 0.0

        bricks = np.zeros((MAX_ROWS, COLUMNS), dtype=np.float32)
        if self.see_bricks:
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
            + self.paddle_hit_reward * returns
            - self.frame_cost
        )

        if bricks > 0:
            self.last_brick_frame = game.frames
        stuck = game.frames - self.last_brick_frame >= STUCK_FRAMES
        out_of_time = (
            self.max_frames is not None and game.frames >= self.max_frames
        )

        terminated = game.game_over
        truncated = not terminated and (stuck or out_of_time)

        info = self.information()
        if terminated:
            info["ending"] = "lost"
        elif stuck:
            info["ending"] = "stuck"
        elif out_of_time:
            info["ending"] = "time limit"
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
