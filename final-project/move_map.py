"""A map of what the final PPO network does while the ball is falling.

The paddle stays in the middle of the screen, the bricks are level 1's
full wall, and the ball is placed at every point of a grid around the
paddle, falling at 30 degrees to the left or to the right. Each arrow
is the move the network finds most likely there; the darker the arrow,
the surer it is.

Run it with:  uv run move_map.py
"""

import matplotlib.pyplot as plt
import torch

from brickgame.env import BrickEnv
from brickgame.game import LEFT, PADDLE_Y, RIGHT, STAY
from players.policy import load_policy
from training.charts import INK, SOFT_INK, save
from training.progress import RUNS_FOLDER

FINAL = RUNS_FOLDER / "seeds-ppo-2" / "best.pt"
OFFSETS = range(-200, 201, 25)
HEIGHTS = range(120, PADDLE_Y, 40)
ARROWS = {STAY: "·", LEFT: "←", RIGHT: "→"}


def chances_at(network, env, offset, height, moving_right):
    """The network's chances of stay, left and right with the ball
    `offset` pixels from the paddle's center, `height` pixels down."""
    game = env.game
    game.paddle_x = 300 - game.paddle_width / 2
    game.ball_x = 300 + offset
    game.ball_y = height
    game.ball_dx = 3.0 if moving_right else -3.0
    game.ball_dy = 5.2
    observation = torch.as_tensor(env.observe()[None])
    with torch.no_grad():
        chances, _ = network(observation)
    return chances.probs[0].tolist()


def main():
    network = load_policy(FINAL).network
    env = BrickEnv()
    env.reset(seed=200_000)

    figure, panels = plt.subplots(2, 1, figsize=(7.5, 7))
    figure.subplots_adjust(
        left=0.2, right=0.92, top=0.86, bottom=0.07, hspace=0.35
    )
    figure.text(
        0.03,
        0.95,
        "What the final player does with a falling ball",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.03,
        0.91,
        "Each arrow is its most likely move; a dot means stay still. "
        "Darker arrows are surer.",
        fontsize=10.5,
        color=SOFT_INK,
    )
    for panel, moving_right in zip(panels, (False, True), strict=True):
        direction = "right" if moving_right else "left"
        panel.set_title(
            f"Ball falling and moving {direction}",
            loc="left",
            fontsize=11.5,
        )
        for column, offset in enumerate(OFFSETS):
            for row, height in enumerate(HEIGHTS):
                chances = chances_at(
                    network, env, offset, height, moving_right
                )
                best = chances.index(max(chances))
                # surest moves in black, even chances in light gray
                shade = 0.8 - 0.8 * (max(chances) - 1 / 3) / (2 / 3)
                panel.text(
                    column,
                    row,
                    ARROWS[best],
                    ha="center",
                    va="center",
                    fontsize=14,
                    color=str(round(shade, 2)),
                )
        panel.set_xlim(-0.5, len(OFFSETS) - 0.5)
        panel.set_ylim(len(HEIGHTS) - 0.5, -0.5)
        panel.set_xticks(
            [0, len(OFFSETS) // 2, len(OFFSETS) - 1],
            ["200 px left", "ball above the paddle", "200 px right"],
        )
        panel.set_yticks(
            [0, len(HEIGHTS) - 1], ["high up", "near the paddle"]
        )
        panel.tick_params(length=0, colors=INK)
        for side in panel.spines.values():
            side.set_visible(False)
    path = save(figure, "move-map.png")
    print(f"map saved to {path}")


if __name__ == "__main__":
    main()
