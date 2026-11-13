"""Compare the four versions of the Deep Q-Network player.

Run it with:  uv run compare_dqn.py
(after training each version, with seed_runs.py for the extra seeds)
"""

from training.charts import versions_compared

# The two versions without a replay memory take a learning step after
# every decision and train several times slower, so they have two
# training runs each instead of three.
VERSIONS = {
    "first version": ["seeds-dqn-first-0", "seeds-dqn-first-1"],
    "target network\nonly": ["seeds-dqn-target-0", "seeds-dqn-target-1"],
    "replay memory\nonly": [f"seeds-dqn-memory-{s}" for s in (0, 1, 2)],
    "memory and\ntarget network": [
        f"seeds-dqn-full-{s}" for s in (0, 1, 2)
    ],
}


def main():
    path = versions_compared(
        VERSIONS, "dqn-versions.png", "What each piece adds"
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
