"""Compare PPO trained with different settings, three runs of 3 million
frames each: first the rewards, then what the player sees and the
learning rate.

Run it with:  uv run compare_settings.py
"""

from training.charts import versions_compared

REWARDS = {
    "normal": "seeds-ppo-normal",
    "no return\nbonus": "seeds-ppo-bonus-0",
    "no frame\ncost": "seeds-ppo-cost-0",
    "return bonus\n1.0": "seeds-ppo-bonus-1",
    "return bonus\n5.0": "seeds-ppo-bonus-5",
}
SEEING = {
    "normal": "seeds-ppo-normal",
    "no paddle\nsize": "seeds-ppo-no-paddle-size",
    "no scaling": "seeds-ppo-no-scaling",
    "no bricks": "seeds-ppo-no-bricks",
    "no scaling,\nno bricks": "seeds-ppo-no-scaling-no-bricks",
    "constant\nlearning rate": "seeds-ppo-constant-rate",
}


def chart(versions, file_name, title):
    runs = {}
    for label, name in versions.items():
        runs[label] = [f"{name}-{seed}" for seed in (0, 1, 2)]
    path = versions_compared(runs, file_name, title, last_tests=4)
    print(f"chart saved to {path}")


def main():
    chart(REWARDS, "rewards-compared.png", "Five rewards")
    chart(SEEING, "seeing-compared.png", "What the player sees")


if __name__ == "__main__":
    main()
