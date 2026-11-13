"""Charts drawn from the run logs in runs/.

Every chart follows the same rules, so a reader who has learned to read
one can read them all:

    the title says what the chart shows, in a plain sentence;
    a line under it says how to read it;
    axes are labeled in words, numbers written as 150,000 or 1.5 million;
    each line carries its name at its right-hand end;
    shades of gray and line styles, so it prints in black and white.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from training.progress import RUNS_FOLDER

CHARTS_FOLDER = Path(__file__).parent.parent / "charts"

INK = "#1f1f1f"
SOFT_INK = "#5c5c5c"
FAINT = "#b5b5b5"
GRID = "#e6e6e6"
SHADES = ["#1f1f1f", "#6b6b6b", "#9a9a9a"]
LINE_STYLES = ["-", "--", ":"]


def read_log(run_name):
    """One dictionary per game, with the numbers turned into numbers."""
    games = []
    with (RUNS_FOLDER / run_name / "log.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            for column, value in row.items():
                if column not in ("ending", "choice"):
                    row[column] = float(value)
            games.append(row)
    return games


def average_bricks(run_name):
    games = read_log(run_name)
    return sum(game["bricks"] for game in games) / len(games)


def moving_average(values, window):
    averages = []
    total = 0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        averages.append(total / min(index + 1, window))
    return averages


def plain_number(value, position=None):
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:g} million"
    if value != int(value):
        return f"{value:g}"
    return f"{value:,.0f}"


def new_chart(title, x_label, y_label, how_to_read=None, label_room=0.05):
    """A blank chart in the book's style, with its title and a line
    saying how to read it. label_room is the share of the width kept
    free on the right for names written at the ends of lines."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelcolor": SOFT_INK,
            "xtick.color": SOFT_INK,
            "ytick.color": SOFT_INK,
        }
    )
    figure, axes = plt.subplots(figsize=(7.5, 4.6))
    figure.subplots_adjust(
        left=0.1, right=1 - label_room, top=0.82, bottom=0.13
    )
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


def label_at_end(axes, y, text, color=INK):
    """Write a line's name just past the right edge of the chart."""
    axes.annotate(
        text,
        xy=(1, y),
        xycoords=("axes fraction", "data"),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        fontsize=10.5,
        color=color,
        annotation_clip=False,
    )


def spread_labels(heights, gap):
    """Nudge label heights apart so no two overlap."""
    placed = list(heights)
    previous = None
    for height, index in sorted(zip(heights, range(len(heights)))):
        if previous is not None and height - previous < gap:
            height = previous + gap
        placed[index] = height
        previous = height
    return placed


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


def returns_over_time(rewards, returns, frames, file_name):
    """The rewards of the first frames as spikes, and the return of
    every frame as a line above them."""
    figure, axes = new_chart(
        "The return rises before every brick",
        "frame",
        "reward and return",
        how_to_read="Spikes are rewards. The line is the return "
        "with discount 0.99.",
    )
    shown = range(1, frames + 1)
    axes.vlines(shown, 0, rewards[:frames], color=FAINT, linewidth=1.2)
    axes.plot(shown, returns[:frames], color=INK, linewidth=1.8)
    axes.axhline(0, color=SOFT_INK, linewidth=0.8)
    axes.set_xlim(0, frames)
    return save(figure, file_name)


def read_tests(run_name):
    path = RUNS_FOLDER / run_name / "tests.csv"
    if not path.exists():
        return []
    with path.open(newline="") as file:
        return [
            {column: float(value) for column, value in row.items()}
            for row in csv.DictReader(file)
        ]


def runs_compared(runs, file_name, title, how_to_read, baselines=()):
    """Test scores of several runs on one chart. runs maps each run's
    folder name to the label written at the end of its line."""
    longest = max(len(label) for label in runs.values())
    figure, axes = new_chart(
        title,
        "frames played while training",
        "average bricks in test games",
        how_to_read=how_to_read,
        label_room=max(0.2, 0.013 * longest + 0.03),
    )
    names = []
    heights = []
    for number, (run_name, label) in enumerate(runs.items()):
        tests = read_tests(run_name)
        frames = [test["frames_played"] for test in tests]
        averages = [test["average"] for test in tests]
        axes.plot(
            frames,
            averages,
            color=SHADES[number % len(SHADES)],
            linestyle=LINE_STYLES[number % len(LINE_STYLES)],
            linewidth=1.8,
        )
        names.append(label)
        heights.append(averages[-1])

    for baseline in baselines:
        average = average_bricks(baseline)
        axes.axhline(average, color=FAINT, linewidth=1)
        names.append(f"{baseline}: {average:,.0f}")
        heights.append(average)

    axes.set_ylim(bottom=0)
    axes.xaxis.set_major_formatter(FuncFormatter(plain_number))
    axes.xaxis.set_major_locator(MaxNLocator(5))
    gap = axes.get_ylim()[1] * 0.06
    for name, height in zip(
        names, spread_labels(heights, gap), strict=True
    ):
        label_at_end(axes, height, name)
    return save(figure, file_name)
