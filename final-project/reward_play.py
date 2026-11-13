"""How the players trained with each reward play: the best network of
every run plays the ten test games, and the totals show how many bricks
it broke, how many times it returned the ball, and how long it lasted.

Run it with:  uv run reward_play.py
"""

import torch

from brickgame.env import BrickEnv
from players import play_game
from players.policy import load_policy
from training.progress import RUNS_FOLDER

VERSIONS = {
    "normal": "seeds-ppo-normal",
    "no return bonus": "seeds-ppo-bonus-0",
    "no frame cost": "seeds-ppo-cost-0",
    "return bonus 1.0": "seeds-ppo-bonus-1",
    "return bonus 5.0": "seeds-ppo-bonus-5",
}
GAMES = 10


def main():
    torch.set_num_threads(1)
    env = BrickEnv()
    print("                  bricks  returns  returns per brick   frames")
    for label, name in VERSIONS.items():
        bricks = returns = frames = 0
        for seed in (0, 1, 2):
            folder = RUNS_FOLDER / f"{name}-{seed}"
            player = load_policy(folder / "best.pt")
            for game in range(GAMES):
                info = play_game(player, env, seed=100_000 + game)
                bricks += info["score"]
                returns += env.game.paddle_hits
                frames += env.game.frames
        games = 3 * GAMES
        print(
            f"{label:16} {bricks / games:7.1f} {returns / games:8.1f}"
            f" {returns / bricks:18.2f} {frames / games:8,.0f}"
        )


if __name__ == "__main__":
    main()
