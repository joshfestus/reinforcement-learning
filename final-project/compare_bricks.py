"""Compare the Deep Q-Network that sees the bricks with the one that
sees only the ball and paddle, three training runs each.

Run it with:  uv run compare_bricks.py
"""

from training.charts import versions_compared

VERSIONS = {
    "sees ball, paddle\nand bricks": [
        f"seeds-dqn-full-{seed}" for seed in (0, 1, 2)
    ],
    "sees ball and\npaddle only": [
        f"seeds-dqn-nobricks-{seed}" for seed in (0, 1, 2)
    ],
}


def main():
    path = versions_compared(
        VERSIONS,
        "dqn-bricks.png",
        "Seeing the bricks makes it learn less, so far",
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
