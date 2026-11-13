# Brick Breaker RL

The source code for *Reinforcement Learning with Python for Beginners* by
Josh Festus: a brick breaker game written from scratch in Python, and the
players that learn to play it, from a table of numbers to PPO.

## Getting started

1. Install uv, as Chapter 3 of the book shows (Appendix A has help if the
   install goes wrong).
2. Download this repository: press the green Code button, choose Download
   ZIP, and unzip it anywhere. Or, with Git: `git clone` this address.
3. Open a terminal in any chapter's folder and run its programs with
   `uv run`. For example:

        cd chapters/ch13-table-of-good-moves
        uv run train_qtable.py

The first `uv run` in a folder downloads Python 3.14 and the libraries that
chapter needs, at the exact versions the book used. Nothing else needs to
be installed.

## How this repository is organized

    chapters/        one folder per chapter, named after it
    final-project/   everything, as it stands at the end of the book
    extras/          in-between steps, deliberate mistakes, extra seed runs

**chapters/** Each folder is a complete copy of your `brick-breaker` project
folder as it stands at the end of that chapter. It has its own
`pyproject.toml`, so every command the chapter prints runs inside its
folder exactly as shown. To start the book at any chapter, copy the folder
of the chapter before it.

**final-project/** The finished game with every player, trainer and chart
program in one place. Use it to play, watch or train anything without
following the chapters in order:

    cd final-project
    uv run play.py                 play the game with the arrow keys
    uv run watch.py chaser         watch a player that never learns
    uv run train_qtable.py         train the table player, about 5 minutes
    uv run train_ppo.py            train the PPO player, about 16 minutes
    uv run watch_ai.py             watch the final trained player

**extras/** Some chapters build a program in several steps, or break it on
purpose so you can read the error. Those versions are here, in a folder
named after the chapter, as `step1`, `step2` and so on, or `broken`. The
`seed_runs.py` helpers trained the extra runs the book uses to check that a
result holds over several seeds.

**Trained players.** Every training run the book quotes is in the `runs`
folder of the chapters that use it: `log.csv` has every training game,
`tests.csv` every test, and `best.pt` (or `best.zip` for the library's PPO)
the best network. You can test, watch and chart them without training
anything yourself.

## The chapters

| Folder | What it builds | Programs it adds |
|---|---|---|
| `ch03-setting-up` | An empty game window | `hello.py`, `sizes.py`, `window.py` |
| `ch04-paddle` | A paddle you steer with the arrow keys | `play.py` |
| `ch05-ball` | A ball that bounces off the walls and the paddle | changes `play.py` |
| `ch06-bricks` | A one level game with bricks, a score and lives | changes `play.py` |
| `ch07-levels` | The finished game, with six levels drawn in text files | `levels/` |
| `ch08-rules-and-screen` | The rules and the drawing in separate files, and a game with no window | `screenshots.py`, `time_game.py` |
| `ch09-environment` | The game as an environment an AI can play | `check_env.py`, `try_env.py` |
| `ch10-players-that-never-learn` | A random, a chasing and a predicting player, measured | `benchmark.py`, `watch.py`, `aim.py` |
| `ch11-rewards-and-looking-ahead` | A recorded game's rewards and their returns | `record_game.py`, `returns.py` |
| `ch12-explore-or-exploit` | A bandit that learns which player is best | `train_bandit.py`, `compare_bandits.py` |
| `ch13-table-of-good-moves` | A Q-table player, about 70 bricks a game | `train_qtable.py`, `compare_fixes.py` |
| `ch14-watching-the-table` | Training reports, a map of the table, SARSA | `report.py`, `draw_table.py`, `replay_table.py`, `compare_sarsa.py`, `table_limits.py` |
| `ch15-network-from-scratch` | A neural network in NumPy that predicts where the ball lands | `record_landings.py`, `train_landing_net.py`, `landing_errors.py` |
| `ch16-network-in-pytorch` | The same network in PyTorch, with Adam | `landing_games.py` |
| `ch17-table-to-network` | A first Deep Q-Network | `train_dqn.py` |
| `ch18-memory-and-target` | The replay memory and the target network, measured | `compare_dqn.py` |
| `ch19-training-the-dqn` | The full DQN run, read closely | `film_strip.py`, `dqn_play.py`, `compare_bricks.py` |
| `ch20-learning-which-move` | A policy network trained with REINFORCE | `train_reinforce.py`, `sample_or_greedy.py` |
| `ch21-actor-and-critic` | An actor and critic on eight games at once | `train_actor_critic.py` |
| `ch22-writing-ppo` | PPO written in full, about 93 bricks a game | `train_ppo.py`, `compare_policies.py` |
| `ch23-ppo-library` | The same game trained with Stable-Baselines3's PPO | `train_library_ppo.py`, `compare_ppo.py` |
| `ch24-fixing-training` | Ten versions of PPO measured three times each, and a guide to failures | `compare_settings.py`, `reward_play.py` |
| `ch25-final-ai` | A fair test of every player, a map of what PPO learned, and watching it play | `final_test.py`, `final_chart.py`, `move_map.py`, `final_film.py`, `watch_ai.py` |
| `ch26-taking-it-further` | Deciding every 2 frames instead of 4, measured, and where to go next | `compare_rhythm.py`, `rhythm_test.py` |

Chapters 1 and 2 have no code.

## Drawing your own level

A level file in `levels/` has up to 6 lines of exactly 8 characters. A dot
is a gap and each of the letters P, B, R, G, O, Y is a brick, from darkest
to lightest:

    ...PP...
    ..BBBB..
    .GGGGGG.

Files are played in name order, so `07-mine.txt` comes after
`06-fortress.txt`.

## License

The code is released under the MIT license.
