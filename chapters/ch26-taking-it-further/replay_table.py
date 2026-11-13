"""Watch a saved Q-table play, in a window.

    uv run replay_table.py runs/qtable/table-1000.json

Plus and minus change the speed, Escape quits.
"""

import sys
from pathlib import Path

from brickgame.env import BrickEnv
from brickgame.render import Screen
from players.qtable import QTablePlayer
from watch import watch


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    table_file = Path(sys.argv[1])
    player = QTablePlayer(epsilon=0)
    player.load(table_file)
    player.name = table_file.stem
    screen = Screen(title=f"Brick Breaker: {table_file}")
    watch(player, BrickEnv(), screen, fps=60)
    screen.close()


if __name__ == "__main__":
    main()
