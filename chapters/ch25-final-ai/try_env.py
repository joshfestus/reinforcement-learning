"""Play the environment with random actions and print what it hands back.

Run it with:  uv run try_env.py
"""

from brickgame.env import BrickEnv

env = BrickEnv()
observation, info = env.reset(seed=1)
env.action_space.seed(1)
print("numbers seen:", len(observation))
print("first seven:", observation[:7].round(3))

total_reward = 0
finished = False
while not finished:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    finished = terminated or truncated
    if abs(reward) > 0.5:
        print(f"frame {info['frames']:5}: reward {reward:+.3f}")

print(f"game over after {info['frames']} frames ({info['ending']})")
print(f"total reward {total_reward:.3f}, score {info['score']}")
