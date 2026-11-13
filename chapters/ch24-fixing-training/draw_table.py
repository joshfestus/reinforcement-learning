"""Draw what a saved Q-table does while the ball is falling.

    uv run draw_table.py runs/qtable/best.json

The chart is saved in the charts folder, named after the table file.
"""

import sys
from pathlib import Path

from training.charts import table_map


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    table_file = Path(sys.argv[1])
    name = f"map-{table_file.parent.name}-{table_file.stem}.png"
    path = table_map(
        table_file, name, "What the table does with a falling ball"
    )
    print(f"map saved to {path}")


if __name__ == "__main__":
    main()
