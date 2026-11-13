"""Compare the three policy players: REINFORCE, actor and critic, and
PPO. One chart follows the test scores of one run of each through
training; the other puts the last tests of all three runs of each side
by side.

Run it with:  uv run compare_policies.py
"""

from training.charts import runs_compared, versions_compared

METHODS = {
    "REINFORCE": "reinforce",
    "actor and critic": "actor-critic",
    "PPO": "ppo",
}


def main():
    first_runs = {}
    for label, method in METHODS.items():
        first_runs[f"seeds-{method}-0"] = label
    path = runs_compared(
        first_runs,
        "policies-over-time.png",
        "PPO learns faster and further",
        "Test scores during one training run of each method.",
    )
    print(f"chart saved to {path}")

    all_runs = {}
    for label, method in METHODS.items():
        all_runs[label] = [f"seeds-{method}-{seed}" for seed in (0, 1, 2)]
    path = versions_compared(
        all_runs, "policies-compared.png", "Three ways to learn a policy"
    )
    print(f"chart saved to {path}")


if __name__ == "__main__":
    main()
