"""Play the environment with random actions and print what it hands back.

Run it with:  uv run try_env.py
"""

import random

from brickgame.env import BrickEnv
from brickgame.game import ACTIONS

env = BrickEnv()
keys = random.Random(1)
observation = env.reset()
print("numbers seen:", len(observation))
print("first seven:", observation[:7].round(3))

total_reward = 0
frames = 0
finished = False
while not finished:
    observation, reward, finished = env.step(keys.choice(ACTIONS))
    total_reward += reward
    frames += 1
    if reward > 0.5 or reward < -0.5:
        print(f"frame {frames:5}: reward {reward:+.3f}")

print(f"game over after {frames} frames")
print(f"total reward {total_reward:.3f}, score {env.game.score}")
