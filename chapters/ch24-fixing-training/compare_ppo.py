"""Compare your PPO with the library's: one run of each through
training, and the last tests of all three runs of each, including the
library's first try, whose learning rate never slid all the way down.

Run it with:  uv run compare_ppo.py
"""

from training.charts import runs_compared, versions_compared

SEEDS = (0, 1, 2)


def main():
    path = runs_compared(
        {"seeds-ppo-0": "your PPO", "seeds-library-ppo-0": "library PPO"},
        "ppo-over-time.png",
        "Two versions of one method",
        "Test scores during one training run of each.",
    )
    print(f"chart saved to {path}")

    path = versions_compared(
        {
            "your PPO": [f"seeds-ppo-{s}" for s in SEEDS],
            "library PPO,\nfirst try": [
                f"seeds-library-ppo-first-{s}" for s in SEEDS
            ],
            "library PPO": [f"seeds-library-ppo-{s}" for s in SEEDS],
        },
        "ppo-compared.png",
        "Your PPO and the library's",
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
