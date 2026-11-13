"""Check the environment against the Gymnasium standard.

Run it with:  uv run check_env.py
"""

from gymnasium.utils.env_checker import check_env

from brickgame.env import BrickEnv

env = BrickEnv()
# The render check only works for environments registered by name with
# Gymnasium, which this project does not need, so it is skipped.
check_env(env, skip_render_check=True)
print("BrickEnv passed every check")
print("actions:", env.action_space)
print("observations:", env.observation_space)
