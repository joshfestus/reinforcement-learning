"""Brick layouts, read from the text files in the levels folder.

Each line of a level file is one row of bricks. A letter is a brick and
picks its color, a dot is a gap. Files are used in name order, and after
the last one the game starts again from the first.
"""

from pathlib import Path

COLUMNS = 8
MAX_ROWS = 6
BRICK_LETTERS = "ROYGBP"
LEVELS_FOLDER = Path(__file__).parent.parent / "levels"


def read_layout(path):
    """Return (row, column, letter) for every brick in a level file."""
    rows = [line.strip() for line in path.read_text().splitlines()]
    rows = [row for row in rows if row]

    if len(rows) > MAX_ROWS:
        raise ValueError(f"{path.name}: more than {MAX_ROWS} rows")

    bricks = []
    for row_number, row in enumerate(rows):
        if len(row) != COLUMNS:
            raise ValueError(
                f"{path.name}, row {row_number + 1}: "
                f"needs {COLUMNS} characters, has {len(row)}"
            )
        for column, letter in enumerate(row):
            if letter == ".":
                continue
            if letter not in BRICK_LETTERS:
                raise ValueError(
                    f"{path.name}, row {row_number + 1}: '{letter}' "
                    f"is not a dot or one of {BRICK_LETTERS}"
                )
            bricks.append((row_number, column, letter))

    if not bricks:
        raise ValueError(f"{path.name}: no bricks")
    return bricks


def level_layout(level):
    """The layout for a level number, counting from 1."""
    files = sorted(LEVELS_FOLDER.glob("*.txt"))
    return read_layout(files[(level - 1) % len(files)])
