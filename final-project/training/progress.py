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
    def __init__(
        self,
        run_name,
        extra=None,
        report_every=10,
        recent=50,
        live_chart=False,
        baselines=(),
    ):
        """extra names one more column a learning player fills in,
        such as "epsilon" or "loss". report_every is how many games pass
        between progress lines; recent is how many of the latest games
        the average covers. live_chart opens a window that redraws the
        learning curve, with the runs named in baselines drawn across."""
        self.folder = RUNS_FOLDER / run_name
        self.folder.mkdir(parents=True, exist_ok=True)
        self.file = (self.folder / "log.csv").open("w", newline="")
        self.writer = csv.writer(self.file)
        header = COLUMNS + [extra] if extra else COLUMNS
        self.writer.writerow(header)
        self.tests_file = None

        self.chart = None
        if live_chart:
            from training.live_chart import LiveChart

            self.chart = LiveChart(run_name, baselines)

        self.report_every = report_every
        self.recent = recent
        self.games = 0
        self.frames_played = 0
        self.scores = []
        self.highest_level = 1
        self.started = time.perf_counter()

    def game_over(self, info, extra=None):
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
        if extra is not None:
            row.append(extra)
        self.writer.writerow(row)

        if self.chart:
            self.chart.add(self.frames_played, info["score"])
        if self.games % self.report_every == 0:
            self.report()

    def test_result(self, scores):
        """Record the scores of test games, played with no exploring."""
        if self.tests_file is None:
            path = self.folder / "tests.csv"
            self.tests_file = path.open("w", newline="")
            self.tests_writer = csv.writer(self.tests_file)
            self.tests_writer.writerow(
                ["games", "frames_played", "average", "best", "worst"]
            )
        average = sum(scores) / len(scores)
        self.tests_writer.writerow(
            [
                self.games,
                self.frames_played,
                round(average, 1),
                max(scores),
                min(scores),
            ]
        )
        self.tests_file.flush()
        print(
            f"test   {len(scores)} games, no exploring: "
            f"{average:.1f} bricks, best {max(scores)}, "
            f"worst {min(scores)}"
        )

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
        if self.tests_file:
            self.tests_file.close()
        if self.chart:
            self.chart.close()
