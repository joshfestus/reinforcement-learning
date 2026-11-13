"""Records every finished game and prints a progress line as it goes.

Each run gets its own folder under runs/, holding log.csv with one row
per game. Every player in the book writes the same columns, so any two
runs can be compared on one chart.
"""

import csv
import time
from pathlib import Path

RUNS_FOLDER = Path(__file__).parent.parent / "runs"

COLUMNS = [
    "game",
    "frames_played",
    "bricks",
    "level",
    "game_frames",
    "lives_lost",
    "ending",
    "seconds",
]


def as_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


class ProgressLog:
    def __init__(self, run_name, report_every=10, recent=50):
        """report_every is how many games pass between progress lines;
        recent is how many of the latest games the average covers."""
        self.folder = RUNS_FOLDER / run_name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = (self.folder / "log.csv").open("w", newline="")
        self.writer = csv.writer(self.file)
        self.writer.writerow(COLUMNS)

        self.report_every = report_every
        self.recent = recent
        self.games = 0
        self.frames_played = 0
        self.scores = []
        self.highest_level = 1
        self.started = time.perf_counter()

    def game_over(self, info):
        self.games += 1
        self.frames_played += info["frames"]
        self.scores.append(info["score"])
        self.highest_level = max(self.highest_level, info["level"])
        seconds = time.perf_counter() - self.started

        row = [
            self.games,
            self.frames_played,
            info["score"],
            info["level"],
            info["frames"],
            info["lives_lost"],
            info["ending"],
            round(seconds, 1),
        ]
        self.writer.writerow(row)

        if self.games % self.report_every == 0:
            self.report()

    def report(self):
        self.file.flush()
        latest = self.scores[-self.recent :]
        average = sum(latest) / len(latest)
        seconds = time.perf_counter() - self.started
        print(
            f"game {self.games}  frames {self.frames_played:,}  "
            f"bricks {average:.1f}  best {max(self.scores)}  "
            f"level {self.highest_level}  {as_time(seconds)}"
        )

    def close(self):
        if self.games % self.report_every != 0:
            self.report()
        self.file.close()
