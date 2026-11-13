"""Six players, the same new game, the same moment: the screen of each
3,000 frames in, or when its game ended if that came first.

Run it with:  uv run final_film.py
"""

import matplotlib.pyplot as plt
import pygame
import torch

from brickgame.env import BrickEnv
from brickgame.render import Screen
from final_test import load_table
from players import PLAYERS
from players.dqn import load_dqn
from players.policy import load_policy
from training.charts import INK, SOFT_INK, save
from training.progress import RUNS_FOLDER

GAME_SEED = 200_000
FRAMES = 3000


def players():
    """The six players, in the order the book met them."""
    return {
        "random": PLAYERS["random"](),
        "chaser": PLAYERS["chaser"](),
        "predictor": PLAYERS["predictor"](),
        "Q-table": load_table(
            RUNS_FOLDER / "seeds-qtable-1" / "best.json"
        ),
        "DQN": load_dqn(RUNS_FOLDER / "seeds-dqn-full-0" / "best.pt"),
        "PPO": load_policy(RUNS_FOLDER / "seeds-ppo-2" / "best.pt"),
    }


def screen_at(player):
    """Play the game for up to FRAMES frames. Returns a picture of the
    screen, the score, and whether the game had ended."""
    env = BrickEnv()
    screen = Screen(window=False)
    observation, info = env.reset(seed=GAME_SEED)
    ended = False
    for _ in range(FRAMES):
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
    torch.set_num_threads(1)
    figure, panels = plt.subplots(2, 3, figsize=(10, 6.2))
    figure.subplots_adjust(
        left=0.01, right=0.99, top=0.86, bottom=0.01, wspace=0.05
    )
    figure.text(
        0.01,
        0.95,
        f"Six players, the same new game, {FRAMES:,} frames in",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    for panel, (name, player) in zip(panels.flat, players().items()):
        picture, score, ended = screen_at(player)
        panel.imshow(picture)
        panel.set_xticks([])
        panel.set_yticks([])
        note = f"game over at {score}" if ended else f"{score} bricks"
        panel.set_title(f"{name}\n{note}", fontsize=10, color=SOFT_INK)
    path = save(figure, "final-film.png")
    print(f"film saved to {path}")


if __name__ == "__main__":
    main()
