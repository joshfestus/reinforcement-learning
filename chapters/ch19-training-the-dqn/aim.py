"""Measure the angles the hand-written players send the ball off at.

Run it with:  uv run aim.py
"""

import math
import statistics

from brickgame.env import BrickEnv
from players import Chaser, Predictor

GAMES = 20


def return_angles(player, env):
    """The angle from straight up, in degrees, of every return the
    player makes in GAMES games."""
    angles = []
    for seed in range(1, GAMES + 1):
        observation, info = env.reset(seed=seed)
        game = env.game
        hits = game.paddle_hits
        finished = False
        while not finished:
            action = player.act(observation)
            observation, reward, terminated, truncated, info = env.step(
                action
            )
            finished = terminated or truncated
            if game.paddle_hits != hits:
                hits = game.paddle_hits
                angle = math.atan2(abs(game.ball_dx), -game.ball_dy)
                angles.append(math.degrees(angle))
    return angles


env = BrickEnv()
for player in [Chaser(), Predictor()]:
    angles = return_angles(player, env)
    steep = sum(angle > 30 for angle in angles) / len(angles)
    print(
        f"{player.name:10} {len(angles):5} returns, "
        f"middle angle {statistics.median(angles):4.1f} degrees, "
        f"steeper than 30: {steep:.1%}"
    )
