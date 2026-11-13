"""Test every player in the book on the same 100 new games.

The games use seeds 200,000 to 200,099, which no training run and no
test during training ever played, so no network has been chosen for
doing well on them. Every learning method is tested with the best
network of each of its three training runs. The results of every game
go into runs/final-test.csv, which final_chart.py draws.

Run it with:  uv run final_test.py
It takes about twenty minutes.
"""

import csv
import statistics

import torch
from stable_baselines3 import PPO

from brickgame.env import BrickEnv
from players import PLAYERS, play_game
from players.dqn import load_dqn
from players.policy import load_policy
from players.qtable import QTablePlayer
from train_library_ppo import LibraryPlayer
from training.progress import RUNS_FOLDER

GAMES = 100
FIRST_SEED = 200_000


def load_table(path):
    player = QTablePlayer(epsilon=0)
    player.load(path)
    return player


def load_library(path):
    return LibraryPlayer(PPO.load(path))


# Each learning method: how to load a saved player, the runs to test,
# and the name of the saved file.
LEARNERS = {
    "Q-table": (load_table, "seeds-qtable", (1, 2, 3), "best.json"),
    "DQN": (load_dqn, "seeds-dqn-full", (0, 1, 2), "best.pt"),
    "REINFORCE": (load_policy, "seeds-reinforce", (0, 1, 2), "best.pt"),
    "actor-critic": (
        load_policy,
        "seeds-actor-critic",
        (0, 1, 2),
        "best.pt",
    ),
    "PPO": (load_policy, "seeds-ppo", (0, 1, 2), "best.pt"),
    "library PPO": (
        load_library,
        "seeds-library-ppo",
        (0, 1, 2),
        "best.zip",
    ),
}


def play_all(player, env):
    """The 100 games, as (bricks, level reached, frames, ending)."""
    games = []
    for game in range(GAMES):
        info = play_game(player, env, seed=FIRST_SEED + game)
        games.append(
            (info["score"], info["level"], info["frames"], info["ending"])
        )
    return games


def main():
    torch.set_num_threads(1)
    env = BrickEnv()
    rows = []
    for name, make_player in PLAYERS.items():
        for game in play_all(make_player(), env):
            rows.append((name, name, *game))
    for name, (load, prefix, seeds, file) in LEARNERS.items():
        for seed in seeds:
            run = f"{prefix}-{seed}"
            player = load(RUNS_FOLDER / run / file)
            for game in play_all(player, env):
                rows.append((name, run, *game))

    path = RUNS_FOLDER / "final-test.csv"
    with open(path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["player", "run", "bricks", "level", "frames", "ending"]
        )
        writer.writerows(rows)

    print("player          runs  bricks  best game  reached level 3")
    for name in [*PLAYERS, *LEARNERS]:
        games = [row for row in rows if row[0] == name]
        runs = len({row[1] for row in games})
        bricks = [row[2] for row in games]
        level_3 = sum(row[3] >= 3 for row in games) / len(games)
        print(
            f"{name:15} {runs:4} {statistics.mean(bricks):7.1f}"
            f" {max(bricks):10} {level_3:15.0%}"
        )
    print(f"every game saved to {path}")


if __name__ == "__main__":
    main()
