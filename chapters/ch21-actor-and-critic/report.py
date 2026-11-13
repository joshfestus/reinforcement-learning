"""Draw a one-page report of a training run.

    uv run report.py qtable

The report is saved in the charts folder.
"""

import sys

from training.charts import training_report

BASELINES = ["random", "chaser", "predictor"]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    print(f"report saved to {training_report(sys.argv[1], BASELINES)}")


if __name__ == "__main__":
    main()
