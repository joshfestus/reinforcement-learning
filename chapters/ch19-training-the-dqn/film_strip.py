"""Show a Deep Q-Network at several points in its training, side by side.

    uv run film_strip.py seeds-dqn-full-0
    uv run film_strip.py seeds-dqn-full-0 2000

Each saved network plays the same test game, and the screen is taken at
the same moment of it, 6,000 frames in unless another number is given,
or when the game ends if that comes first. The picture is saved in the
charts folder.
"""

import sys

import matplotlib.pyplot as plt
import pygame

from brickgame.env import BrickEnv
from brickgame.render import Screen
from players.dqn import load_dqn
from training.charts import INK, SOFT_INK, save
from training.progress import RUNS_FOLDER

GAME_SEED = 100_000
PICTURES = 4


def checkpoints(run_name):
    """The saved networks of a run, from first to last, as (frames
    trained, file). Four are picked, spread across the run."""
    files = (RUNS_FOLDER / run_name).glob("network-*.pt")
    saved = sorted((int(f.stem.split("-")[1]), f) for f in files)
    last = len(saved) - 1
    picks = [round(n * last / (PICTURES - 1)) for n in range(PICTURES)]
    return [saved[pick] for pick in picks]


def screen_at(player, frames):
    """Play the test game for up to `frames` frames. Returns the picture
    and the score at that moment, and whether the game had ended."""
    env = BrickEnv()
    screen = Screen(window=False)
    observation, info = env.reset(seed=GAME_SEED)
    ended = False
    for _ in range(frames):
        observation, reward, terminated, truncated, info = env.step(
            player.act(observation)
        )
        if terminated or truncated:
            ended = True
            break
    screen.draw(env.game)
    picture = pygame.surfarray.array3d(screen.surface).swapaxes(0, 1)
    return picture, env.game.score, ended


def main():
    if len(sys.argv) not in (2, 3):
        print(__doc__)
        sys.exit(1)
    run_name = sys.argv[1]
    frames = int(sys.argv[2]) if len(sys.argv) == 3 else 6000

    figure, panels = plt.subplots(2, PICTURES // 2, figsize=(9, 8))
    figure.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)
    figure.text(
        0.01,
        0.96,
        f"The same test game, {frames:,} frames in, "
        "at four points in training",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    for panel, (trained, path) in zip(panels.flat, checkpoints(run_name)):
        picture, score, ended = screen_at(load_dqn(path), frames)
        panel.imshow(picture)
        panel.set_xticks([])
        panel.set_yticks([])
        note = "game over" if ended else f"{score} bricks so far"
        panel.set_title(
            f"after {trained:,} frames\n{note}",
            fontsize=10,
            color=SOFT_INK,
        )
    path = save(figure, f"film-{run_name}.png")
    print(f"film strip saved to {path}")


if __name__ == "__main__":
    main()
