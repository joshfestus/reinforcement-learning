"""Save pictures of the game without opening a window.

Saves the start of every level, and one moment of play, to the
screenshots folder.

Run it with:  uv run screenshots.py
"""

from pathlib import Path

from brickgame.game import BrickGame
from brickgame.levels import LEVELS_FOLDER
from brickgame.render import Screen

OUTPUT = Path(__file__).parent / "screenshots"


def main():
    OUTPUT.mkdir(exist_ok=True)
    screen = Screen(window=False)
    level_count = len(list(LEVELS_FOLDER.glob("*.txt")))

    game = BrickGame(seed=3)
    for level in range(1, level_count + 1):
        game.level = level
        game.start_level()
        screen.draw(game)
        screen.save(OUTPUT / f"level-{level}.png")

    game.reset()
    for _ in range(140):
        game.step(0)
    screen.draw(game)
    screen.save(OUTPUT / "playing.png")

    print(f"saved {level_count + 1} pictures in {OUTPUT}")


if __name__ == "__main__":
    main()
