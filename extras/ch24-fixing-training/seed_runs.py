"""Train train_ppo.py with another seed, under another run name, with
any of its options.

    uv run seed_runs.py FOLDER SEED RUN_NAME [options for train_ppo.py]

For example, from the chapter's folder:
    uv run ../../extras/ch24-fixing-training/seed_runs.py . 1 \\
        seeds-ppo-bonus-0-1 --return-bonus 0 --frames 3000000

The seed sets the network's starting weights and the player's random
choices. The games keep their usual seeds.
"""

import importlib
import os
import sys

folder, seed, run_name = sys.argv[1:4]
options = sys.argv[4:]
seed = int(seed)
sys.path.insert(0, os.path.abspath(folder))
os.chdir(folder)
module = importlib.import_module("train_ppo")

make_log = module.ProgressLog
module.ProgressLog = lambda name, **settings: make_log(
    run_name, **settings
)
make_player = module.PolicyPlayer
module.PolicyPlayer = lambda **settings: make_player(
    **{**settings, "seed": seed}
)
sys.argv = ["train_ppo", *options]
module.main()
