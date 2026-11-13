"""Train one of the policy trainers with another seed, under another run
name.

    uv run seed_runs.py FOLDER TRAINER SEED RUN_NAME

TRAINER is train_reinforce, train_actor_critic, train_ppo or
train_library_ppo. For example, from the final-project folder:
    uv run ../extras/ch20-learning-which-move/seed_runs.py . \
        train_reinforce 1 seeds-reinforce-1

The seed sets the network's starting weights and the player's random
choices. The games keep their usual seeds.
"""

import importlib
import os
import sys

folder, trainer, seed, run_name = sys.argv[1:5]
seed = int(seed)
sys.path.insert(0, os.path.abspath(folder))
os.chdir(folder)
module = importlib.import_module(trainer)

make_log = module.ProgressLog
module.ProgressLog = lambda name, **settings: make_log(
    run_name, **settings
)
if hasattr(module, "PolicyPlayer"):
    make_player = module.PolicyPlayer
    module.PolicyPlayer = lambda **settings: make_player(
        **{**settings, "seed": seed}
    )
if hasattr(module, "PPO"):
    make_ppo = module.PPO
    module.PPO = lambda *args, **settings: make_ppo(
        *args, **{**settings, "seed": seed}
    )
sys.argv = [trainer]
module.main()
