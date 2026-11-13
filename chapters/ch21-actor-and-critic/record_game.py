"""Play one game with the chaser and save the reward of every frame.

Run it with:  uv run record_game.py
"""

import csv

from brickgame.env import BrickEnv
from players import Chaser
from training.progress import RUNS_FOLDER

SEED = 1


def main():
    env = BrickEnv()
    player = Chaser()
    observation, info = env.reset(seed=SEED)
    rewards = []
    finished = False
    while not finished:
        action = player.act(observation)
        observation, reward, terminated, truncated, info = env.step(
            action
        )
        rewards.append(reward)
        finished = terminated or truncated

    folder = RUNS_FOLDER / "one-game"
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "rewards.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["frame", "reward"])
        for frame, reward in enumerate(rewards, start=1):
            writer.writerow([frame, round(reward, 6)])

    print(
        f"{len(rewards):,} frames, {info['score']} bricks, "
        f"{info['lives_lost']} lives lost, level {info['level']}"
    )
    print(f"total reward {sum(rewards):.3f}")
    print("the first large rewards:")
    shown = 0
    for frame, reward in enumerate(rewards, start=1):
        if abs(reward) > 0.01 and shown < 6:
            print(f"  frame {frame:4}: {reward:+.3f}")
            shown += 1


if __name__ == "__main__":
    main()
