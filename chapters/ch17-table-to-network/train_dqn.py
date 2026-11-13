"""Train the first Deep Q-Network player.

It learns from each decision as soon as it is made. Every 250,000 frames
it plays 10 test games without exploring; the network is saved each
time, and the best one is kept as best.pt.

Run it with:  uv run train_dqn.py
Add --live to watch the learning curve, --frames N to train for N
frames instead of 6 million.
"""

import sys

import numpy as np
import torch

from brickgame.env import BrickEnv
from players import play_game
from players.dqn import DQNPlayer
from training.decisions import play_decision, sliding
from training.progress import ProgressLog

TOTAL_FRAMES = 6_000_000
TRAINING_FRAMES = 10_000
TEST_EVERY = 250_000
TEST_GAMES = 10

START_EPSILON = 1.0
END_EPSILON = 0.05
EXPLORING_SHARE = 0.25


def test(player, games):
    saved_epsilon = player.epsilon
    player.epsilon = 0
    env = BrickEnv()
    scores = []
    for game in range(games):
        scores.append(
            play_game(player, env, seed=100_000 + game)["score"]
        )
    player.epsilon = saved_epsilon
    return scores


def main():
    # The network is small, so one processor thread is faster than
    # several that spend their time waiting for each other.
    torch.set_num_threads(1)
    total_frames = TOTAL_FRAMES
    if "--frames" in sys.argv:
        total_frames = int(sys.argv[sys.argv.index("--frames") + 1])

    player = DQNPlayer()
    env = BrickEnv(max_frames=TRAINING_FRAMES)
    log = ProgressLog(
        "dqn-first",
        extra="loss",
        report_every=20,
        live_chart="--live" in sys.argv,
    )
    player.save(log.folder / "network-0.pt")

    frames = 0
    next_test = TEST_EVERY
    best_average = -1
    game = 0
    while frames < total_frames:
        game += 1
        observation, info = env.reset(seed=game)
        state = player.state(observation)
        losses = []
        finished = False
        while not finished:
            exploring = frames / (total_frames * EXPLORING_SHARE)
            player.epsilon = sliding(
                START_EPSILON, END_EPSILON, exploring
            )
            action = player.choose(state)

            observation, reward, played, finished, ended, info = (
                play_decision(env, action, player.decide_every)
            )
            frames += played
            new_state = player.state(observation)
            losses.append(
                player.learn(state, action, reward, new_state, ended)
            )
            state = new_state

        log.game_over(info, extra=round(float(np.mean(losses)), 5))

        if frames >= next_test:
            next_test += TEST_EVERY
            scores = test(player, TEST_GAMES)
            log.test_result(scores)
            player.save(log.folder / f"network-{frames}.pt")
            average = sum(scores) / len(scores)
            if average > best_average:
                best_average = average
                player.save(log.folder / "best.pt")
                print(f"       new best network: {average:.1f} bricks")

    log.close()
    print(f"\nbest network saved to {log.folder / 'best.pt'}")


if __name__ == "__main__":
    main()
