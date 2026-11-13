"""Deciding every 4 frames, every 2 frames, and every 2 frames looking
further ahead: three runs of 3 million frames each.

Run it with:  uv run compare_rhythm.py
"""

from training.charts import versions_compared

VERSIONS = {
    "every 4 frames": "seeds-ppo-normal",
    "every 2 frames": "seeds-ppo-every-2",
    "every 2 frames,\ndiscount 0.995": "seeds-ppo-every-2-discount",
}


def main():
    runs = {}
    for label, name in VERSIONS.items():
        runs[label] = [f"{name}-{seed}" for seed in (0, 1, 2)]
    path = versions_compared(
        runs, "rhythm-compared.png", "Deciding more often", last_tests=4
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
