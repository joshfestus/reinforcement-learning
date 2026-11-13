"""Compare the four versions of the Deep Q-Network player.

Run it with:  uv run compare_dqn.py
(after training each version, with seed_runs.py for the extra seeds)
"""

from training.charts import versions_compared

# The two versions without a replay memory take a learning step after
# every decision and train several times slower. They have two training
# runs each instead of three, stopped after 3.75 million frames, so every
# version is compared on its last five tests up to that point.
UP_TO_FRAMES = 3_800_000
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
        VERSIONS,
        "dqn-versions.png",
        "The replay memory is what makes the difference",
        last_tests=5,
        up_to_frames=UP_TO_FRAMES,
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
