"""Does deciding more often help? Test PPO players that decided every 4
frames and every 2 frames, three runs of 3 million frames each, on
Chapter 25's 100 new games.

Run it with:  uv run rhythm_test.py
It takes about fifteen minutes.
"""

import statistics

import torch

from brickgame.env import BrickEnv
from final_test import play_all
from players.policy import load_policy
from training.progress import RUNS_FOLDER

VERSIONS = {
    "every 4 frames": "seeds-ppo-normal",
    "every 2 frames": "seeds-ppo-every-2",
}


def main():
    torch.set_num_threads(1)
    env = BrickEnv()
    print("version          run   bricks  reached level 4")
    for label, prefix in VERSIONS.items():
        averages = []
        for seed in (0, 1, 2):
            player = load_policy(
                RUNS_FOLDER / f"{prefix}-{seed}" / "best.pt"
            )
            games = play_all(player, env)
            bricks = statistics.mean(game[0] for game in games)
            level_4 = sum(game[1] >= 4 for game in games) / len(games)
            averages.append(bricks)
            print(f"{label:16} {seed:3} {bricks:8.1f} {level_4:16.0%}")
        print(f"{label:16} all {statistics.mean(averages):8.1f}")


if __name__ == "__main__":
    main()
