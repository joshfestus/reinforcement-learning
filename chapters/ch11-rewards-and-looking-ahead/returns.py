"""Work out the returns of the recorded game for three discounts.

Run it with:  uv run returns.py   (after record_game.py)
"""

import csv

from training.charts import returns_over_time
from training.progress import RUNS_FOLDER
from training.returns import discounted_returns

DISCOUNTS = [0.9, 0.99, 0.999]
FRAMES_SHOWN = [100, 105, 108, 109, 110, 130, 150, 155, 158, 159, 160]


def main():
    path = RUNS_FOLDER / "one-game" / "rewards.csv"
    with path.open(newline="") as file:
        rewards = [float(row["reward"]) for row in csv.DictReader(file)]

    returns = {}
    for discount in DISCOUNTS:
        returns[discount] = discounted_returns(rewards, discount)

    print("frame   reward   return at 0.9   at 0.99   at 0.999")
    for frame in FRAMES_SHOWN:
        index = frame - 1
        print(
            f"{frame:5}   {rewards[index]:+.3f}   "
            f"{returns[0.9][index]:13.3f}   "
            f"{returns[0.99][index]:7.3f}   "
            f"{returns[0.999][index]:8.3f}"
        )

    chart = returns_over_time(rewards, returns[0.99], 400, "returns.png")
    print(f"chart saved to {chart}")


if __name__ == "__main__":
    main()
