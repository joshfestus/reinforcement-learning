"""Every player in the book on one chart, from the fair test that
final_test.py saved.

Run it with:  uv run final_chart.py
(after final_test.py)
"""

import csv
import statistics

from training.charts import averages_compared
from training.progress import RUNS_FOLDER

LABELS = {
    "random": "random",
    "chaser": "chaser",
    "predictor": "predictor",
    "Q-table": "Q-table",
    "DQN": "DQN",
    "REINFORCE": "REIN-\nFORCE",
    "actor-critic": "actor and\ncritic",
    "PPO": "PPO",
    "library PPO": "library\nPPO",
}


def main():
    with open(RUNS_FOLDER / "final-test.csv") as file:
        rows = list(csv.DictReader(file))
    groups = {}
    for name, label in LABELS.items():
        runs = sorted(
            {row["run"] for row in rows if row["player"] == name}
        )
        groups[label] = [
            statistics.mean(
                int(row["bricks"]) for row in rows if row["run"] == run
            )
            for run in runs
        ]
    path = averages_compared(
        groups,
        "final-players.png",
        "Every player in the book",
        "100 new games each. A dot is one training run; the bar is the "
        "average.",
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
