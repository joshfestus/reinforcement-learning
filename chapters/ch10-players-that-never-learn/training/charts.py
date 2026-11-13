"""Charts drawn from the run logs in runs/.

Every chart follows the same rules, so a reader who has learned to read
one can read them all:

    the title says what the chart shows, in a plain sentence;
    a line under it says how to read it;
    axes are labeled in words, numbers written as 150,000 or 1.5 million;
    shades of gray and line styles, so it prints in black and white.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from training.progress import RUNS_FOLDER

CHARTS_FOLDER = Path(__file__).parent.parent / "charts"

INK = "#1f1f1f"
SOFT_INK = "#5c5c5c"
FAINT = "#b5b5b5"
GRID = "#e6e6e6"


def read_log(run_name):
    """One dictionary per game, with the numbers turned into numbers."""
    games = []
    with (RUNS_FOLDER / run_name / "log.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            for column, value in row.items():
                if column != "ending":
                    row[column] = float(value)
            games.append(row)
    return games


def plain_number(value, position=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g} million"
    if value != int(value):
        return f"{value:g}"
    return f"{value:,.0f}"


def new_chart(title, x_label, y_label, how_to_read=None):
    """A blank chart in the book's style, with its title and a line
    saying how to read it."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelcolor": SOFT_INK,
            "xtick.color": SOFT_INK,
            "ytick.color": SOFT_INK,
        }
    )
    figure, axes = plt.subplots(figsize=(7.5, 4.6))
    figure.subplots_adjust(left=0.1, right=0.95, top=0.82, bottom=0.13)
    figure.text(
        0.1, 0.94, title, fontsize=14, fontweight="bold", color=INK
    )
    if how_to_read:
        figure.text(
            0.1, 0.885, how_to_read, fontsize=10.5, color=SOFT_INK
        )

    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(FAINT)
    axes.tick_params(length=0)
    axes.grid(axis="y", color=GRID, linewidth=0.8)
    axes.set_axisbelow(True)
    axes.yaxis.set_major_formatter(FuncFormatter(plain_number))
    return figure, axes


def save(figure, file_name):
    CHARTS_FOLDER.mkdir(exist_ok=True)
    path = CHARTS_FOLDER / file_name
    figure.savefig(path, dpi=200)
    plt.close(figure)
    return path


def players_compared(run_names, file_name, title):
    """One column per player: every game as a dot, the average as a bar
    across the column, with the average written beside it."""
    figure, axes = new_chart(
        title,
        "",
        "bricks broken in one game",
        how_to_read="Each dot is one game. "
        "The black bar is the player's average.",
    )
    for column, run_name in enumerate(run_names):
        bricks = [game["bricks"] for game in read_log(run_name)]
        average = sum(bricks) / len(bricks)

        # Spread the dots sideways a little, in a fixed pattern, so
        # games with the same score do not hide behind each other.
        sideways = []
        for game in range(len(bricks)):
            sideways.append(column + ((game * 37) % 41 - 20) / 80)
        axes.scatter(
            sideways,
            bricks,
            s=14,
            facecolors="none",
            edgecolors=FAINT,
            linewidths=0.8,
        )
        axes.plot(
            [column - 0.32, column + 0.32],
            [average, average],
            color=INK,
            linewidth=3,
            solid_capstyle="butt",
        )
        axes.annotate(
            f"{average:,.0f}",
            xy=(column + 0.36, average),
            va="center",
            fontsize=12,
            fontweight="bold",
            color=INK,
        )

    axes.set_xticks(range(len(run_names)), run_names, fontsize=12)
    axes.set_xlim(-0.6, len(run_names) - 0.2)
    axes.set_ylim(bottom=0)
    axes.spines["bottom"].set_visible(False)
    return save(figure, file_name)
