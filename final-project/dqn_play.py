"""Measure how the best trained Deep Q-Networks play: bricks, lives lost,
how games end, and the angles they send the ball off at.

Run it with:  uv run dqn_play.py
"""

import math
import statistics
from collections import Counter

from brickgame.env import BrickEnv
from players.dqn import load_dqn
from training.progress import RUNS_FOLDER

GAMES = 20
RUNS = [f"seeds-dqn-full-{s}" for s in (0, 1, 2)] + [
    f"seeds-dqn-nobricks-{s}" for s in (0, 1, 2)
]


def measure(player):
    env = BrickEnv()
    angles, scores, lives, endings = [], [], 0, Counter()
    for game in range(GAMES):
        observation, info = env.reset(seed=100_000 + game)
        hits = env.game.paddle_hits
        finished = False
        while not finished:
            action = player.act(observation)
            observation, reward, terminated, truncated, info = env.step(
                action
            )
            finished = terminated or truncated
            if env.game.paddle_hits != hits:
                hits = env.game.paddle_hits
                ball = env.game
                angle = math.atan2(abs(ball.ball_dx), -ball.ball_dy)
                angles.append(math.degrees(angle))
        scores.append(info["score"])
        lives += info["lives_lost"]
        endings[info["ending"]] += 1
    return angles, scores, lives / GAMES, endings


def main():
    for run_name in RUNS:
        player = load_dqn(RUNS_FOLDER / run_name / "best.pt")
        angles, scores, lives, endings = measure(player)
        steep = sum(angle > 30 for angle in angles) / len(angles)
        name = run_name.removeprefix("seeds-dqn-")
        print(
            f"{name:10} {statistics.mean(scores):6.1f} bricks, "
            f"{lives:.1f} lives lost, {endings['stuck']} stuck, "
            f"angle {statistics.median(angles):4.1f}, steep {steep:.0%}"
        )


if __name__ == "__main__":
    main()
