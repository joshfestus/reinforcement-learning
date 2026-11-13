"""Compare three ways of exploring, over 1,000 runs each.

Playing 1,000 runs of 200 real games would take hours, so each game here
is replaced by a score drawn at random from the 100 games benchmark.py
recorded for that player. Run benchmark.py first.

Run it with:  uv run compare_bandits.py
"""

import random

from players import PLAYERS
from players.bandit import Bandit
from training.charts import (
    LINE_STYLES,
    label_at_end,
    new_chart,
    read_log,
    save,
    spread_labels,
)

RUNS = 1000
GAMES = 200
BEST = "predictor"

STRATEGIES = {
    "never explores": {"epsilon": 0.0},
    "explores 1 game in 10": {"epsilon": 0.1},
    "explores less and less": {"epsilon": 1.0, "decay": 0.97},
}


def main():
    names = list(PLAYERS)
    recorded = [
        [game["bricks"] for game in read_log(name)] for name in names
    ]
    draw = random.Random(0)

    figure, axes = new_chart(
        "A learner that never explores often misses the best player",
        "games played",
        f"runs that picked the {BEST} (%)",
        how_to_read=f"Each line is {RUNS:,} runs of one way of choosing.",
        label_room=0.27,
    )
    finals = []
    for number, (label, settings) in enumerate(STRATEGIES.items()):
        chose_best = [0] * GAMES
        for run in range(RUNS):
            bandit = Bandit(names, seed=run, **settings)
            for game in range(GAMES):
                arm = bandit.choose()
                bandit.learn(arm, draw.choice(recorded[arm]))
                if names[arm] == BEST:
                    chose_best[game] += 1

        percent = [100 * count / RUNS for count in chose_best]
        print(f"{label}: chose the {BEST} in {percent[-1]:.0f}% of runs")
        axes.plot(
            range(1, GAMES + 1),
            percent,
            color="#1f1f1f",
            linestyle=LINE_STYLES[number],
            linewidth=1.8,
        )
        finals.append(percent[-1])

    axes.set_ylim(0, 100)
    for label, height in zip(
        STRATEGIES, spread_labels(finals, 6), strict=True
    ):
        label_at_end(axes, height, label)
    print(f"chart saved to {save(figure, 'bandits.png')}")


if __name__ == "__main__":
    main()
