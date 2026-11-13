"""Train one version of train_qtable.py with another seed for the player,
under another run name. Used to repeat each run of Chapter 13 with
three more seeds.

    uv run seed_runs.py FOLDER SEED RUN_NAME

For example, from inside extras/ch13-table-of-good-moves:
    uv run seed_runs.py step1 2 seeds-qtable-first-2

The runs land in that folder's runs folder; the book's copies are in
chapters/ch13-table-of-good-moves/runs.
"""

import os
import sys

folder, seed, run_name = sys.argv[1], int(sys.argv[2]), sys.argv[3]
sys.path.insert(0, os.path.abspath(folder))
os.chdir(folder)
import train_qtable

make_player = train_qtable.QTablePlayer
make_log = train_qtable.ProgressLog
train_qtable.QTablePlayer = lambda **settings: make_player(
    **{**settings, "seed": seed}
)
train_qtable.ProgressLog = lambda name, **settings: make_log(
    run_name, **settings
)
train_qtable.main()
