"""Compare the versions of the Q-table player, four training runs each.

Run it with:  uv run compare_fixes.py
(after training each version, with seed_runs.py for the extra seeds)
"""

from training.charts import versions_compared

VERSIONS = {
    "first\nversion": ["qtable-first"],
    "deciding every\n4 frames": ["qtable-every-4"],
    "and a lost\nlife ends it": ["qtable"],
    "with a\nshrinking rate": ["qtable-shrinking"],
}
EXTRA_RUNS = {
    "qtable-first": "seeds-qtable-first",
    "qtable-every-4": "seeds-qtable-every-4",
    "qtable": "seeds-qtable-lives",
    "qtable-shrinking": "seeds-qtable-shrinking",
}


def main():
    for runs in VERSIONS.values():
        first = runs[0]
        for seed in (1, 2, 3):
            runs.append(f"{EXTRA_RUNS[first]}-{seed}")
    path = versions_compared(
        VERSIONS,
        "qtable-fixes.png",
        "Two changes more than double what the table learns",
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
