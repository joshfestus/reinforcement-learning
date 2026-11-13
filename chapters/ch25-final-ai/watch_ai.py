"""Watch the final PPO player in a window, level after level.

    uv run watch_ai.py

Keys while watching, as in watch.py: + and - change the speed, N starts
a new game, Escape quits.
"""

import torch

from brickgame.env import BrickEnv
from brickgame.render import Screen
from players.policy import load_policy
from training.progress import RUNS_FOLDER
from watch import watch

FINAL = RUNS_FOLDER / "seeds-ppo-2" / "best.pt"


def main():
    torch.set_num_threads(1)
    player = load_policy(FINAL)
    player.name = "final PPO player"
    env = BrickEnv()
    screen = Screen(title="Brick Breaker: the final AI")
    fps = 60
    keep_going = True
    while keep_going:
        fps, keep_going = watch(player, env, screen, fps)
    screen.close()


if __name__ == "__main__":
    main()
