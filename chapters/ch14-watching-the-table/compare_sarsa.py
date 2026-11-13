"""Compare Q-learning and SARSA, four training runs each.

Run it with:  uv run compare_sarsa.py
(after training both, with seed_runs.py for the extra seeds)
"""

from training.charts import versions_compared

VERSIONS = {
    "Q-learning": ["qtable"]
    + [f"seeds-qtable-{seed}" for seed in (1, 2, 3)],
    "SARSA": ["sarsa"] + [f"seeds-sarsa-{seed}" for seed in (1, 2, 3)],
}


def main():
    path = versions_compared(
        VERSIONS,
        "qlearning-sarsa.png",
        "On this game, Q-learning learns more than SARSA",
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
