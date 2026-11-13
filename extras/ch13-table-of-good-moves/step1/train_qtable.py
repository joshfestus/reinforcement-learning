"""Train the Q-table player.

Every 250 games the player plays 20 test games without exploring. Its
table is saved each time, and the best one is kept as best.json. Run
benchmark.py first: the live chart compares the table with those
players.

Run it with:  uv run train_qtable.py
Add --live to watch the learning curve while it trains.
"""

import sys

from brickgame.env import BrickEnv
from players import play_game
from players.qtable import DECIDE_EVERY, QTablePlayer, situation
from training.decisions import play_decision, sliding
from training.progress import ProgressLog

GAMES = 4000
TEST_EVERY = 250
TEST_GAMES = 20

# Training games stop at 5,000 frames, so a player that has learned to
# keep the ball alive still moves on to fresh games and fresh serves.
TRAINING_FRAMES = 5000

START_EPSILON = 1.0
END_EPSILON = 0.02
EXPLORING_SHARE = 0.5


def play_and_learn(player, env, seed):
    """Play one training game, learning after every decision."""
    observation, info = env.reset(seed=seed)
    state = situation(observation)
    action = player.choose(state)
    finished = False
    while not finished:
        observation, reward, played, finished, ended, info = (
            play_decision(env, action, DECIDE_EVERY)
        )
        new_state = situation(observation)
        new_action = player.choose(new_state)
        player.learn(state, action, reward, new_state, finished)
        state, action = new_state, new_action
    return info


def test(player, games):
    """Play test games without exploring. Returns the scores."""
    saved_epsilon = player.epsilon
    player.epsilon = 0
    env = BrickEnv()
    scores = []
    for game in range(games):
        info = play_game(player, env, seed=100_000 + game)
        scores.append(info["score"])
    player.epsilon = saved_epsilon
    return scores


def main():
    player = QTablePlayer(seed=0)
    env = BrickEnv(max_frames=TRAINING_FRAMES)
    log = ProgressLog(
        "qtable-first",
        extra="epsilon",
        report_every=100,
        live_chart="--live" in sys.argv,
        baselines=["random", "chaser", "predictor"],
    )
    player.save(log.folder / "table-0.json")
    best_average = -1

    for game in range(1, GAMES + 1):
        # Explore less as training goes on.
        progress = game / GAMES
        player.epsilon = sliding(
            START_EPSILON, END_EPSILON, progress / EXPLORING_SHARE
        )

        info = play_and_learn(player, env, seed=game)
        log.game_over(info, extra=round(player.epsilon, 3))

        if game % TEST_EVERY == 0:
            scores = test(player, TEST_GAMES)
            log.test_result(scores)
            player.save(log.folder / f"table-{game}.json")
            average = sum(scores) / len(scores)
            if average > best_average:
                best_average = average
                player.save(log.folder / "best.json")
                print(f"       new best table: {average:.1f} bricks")

    log.close()
    print(f"\nbest table saved to {log.folder / 'best.json'}")


if __name__ == "__main__":
    main()
