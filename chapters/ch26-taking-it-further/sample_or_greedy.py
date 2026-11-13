"""Test trained policy players two ways: taking the most likely move,
as the trainer's tests do, and picking moves with their chances, as in
training. Each name given is a method whose three training runs, with
seeds 0, 1 and 2, are tested.

    uv run sample_or_greedy.py reinforce
    uv run sample_or_greedy.py actor-critic ppo
"""

import statistics
import sys

import torch

from brickgame.env import BrickEnv
from players import play_game
from players.policy import load_policy
from training.progress import RUNS_FOLDER

GAMES = 10


def average_score(player, env):
    scores = []
    for game in range(GAMES):
        scores.append(
            play_game(player, env, seed=100_000 + game)["score"]
        )
    return statistics.mean(scores)


def main():
    env = BrickEnv()
    run_names = []
    for method in sys.argv[1:]:
        for seed in (0, 1, 2):
            run_names.append(f"seeds-{method}-{seed}")
    for run_name in run_names:
        player = load_policy(RUNS_FOLDER / run_name / "best.pt")
        torch.manual_seed(0)
        player.greedy = True
        greedy = average_score(player, env)
        player.greedy = False
        sampling = average_score(player, env)
        print(
            f"{run_name:21} most likely: {greedy:5.1f} bricks   "
            f"with chances: {sampling:5.1f}"
        )


if __name__ == "__main__":
    main()
