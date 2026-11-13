"""Train train_dqn.py with another seed, under another run name.

    uv run seed_runs.py FOLDER SEED RUN_NAME [switches for train_dqn.py]

For example, from the final-project folder:
    uv run ../extras/ch17-table-to-network/seed_runs.py . 1 seeds-dqn-1
    uv run ../extras/ch17-table-to-network/seed_runs.py . 1 \
        seeds-dqn-first-1 --no-memory --no-target

The seed sets the network's starting weights, the player's exploring and
which past decisions are drawn from the memory. The games keep their
usual seeds.
"""

import os
import sys

import numpy as np

folder, seed, run_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.abspath(folder))
os.chdir(folder)
import train_dqn

make_player = train_dqn.DQNPlayer
make_log = train_dqn.ProgressLog
make_rng = np.random.default_rng
train_dqn.DQNPlayer = lambda **settings: make_player(
    **{**settings, "seed": seed}
)
train_dqn.ProgressLog = lambda name, **settings: make_log(
    run_name, **settings
)
train_dqn.np.random.default_rng = lambda first_seed: make_rng(seed)
train_dqn.main()
