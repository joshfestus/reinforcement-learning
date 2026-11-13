"""Time the game running with no window, with and without drawing.

Run it with:  uv run time_game.py
"""

import random
import time

from brickgame.game import ACTIONS, BrickGame
from brickgame.render import Screen

FRAMES = 200_000


def frames_per_second(screen=None):
    """Play FRAMES frames with random key presses and return how many
    frames a second that came to. With a screen, draw every frame."""
    game = BrickGame(seed=1)
    keys = random.Random(1)
    start = time.perf_counter()
    for frame in range(FRAMES):
        if game.game_over:
            game.reset()
        game.step(keys.choice(ACTIONS))
        if screen:
            screen.draw(game)
    seconds = time.perf_counter() - start
    return FRAMES / seconds


rules_only = frames_per_second()
with_drawing = frames_per_second(Screen(window=False))

print(f"rules only:        {rules_only:9,.0f} frames a second")
print(f"rules and drawing: {with_drawing:9,.0f} frames a second")
print("played by a person:       60 frames a second")
print(f"the rules alone are {rules_only / 60:,.0f} times faster")
