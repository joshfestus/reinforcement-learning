"""Train the Deep Q-Network player.

Every 250,000 frames it plays 10 test games without exploring; the
network is saved each time, and the best one is kept as best.pt.

Run it with:  uv run train_dqn.py
Switches, to see what each piece does:
    --no-memory   learn only from the latest decision, no replay memory
    --no-target   value the next situation with the network itself
    --no-bricks   see only the ball and paddle, not the bricks
    --live        watch the learning curve while it trains
    --frames N    train for N frames instead of 6 million
    --decide-every N   keep each action for N frames (normally 4)
"""

import sys

import numpy as np
import torch

from brickgame.env import BrickEnv
from players import play_game
from players.dqn import DQNPlayer, ReplayMemory
from training.decisions import play_decision, sliding
from training.progress import ProgressLog

TOTAL_FRAMES = 6_000_000
TRAINING_FRAMES = 10_000
TEST_EVERY = 250_000
TEST_GAMES = 10

MEMORY_SIZE = 100_000
BATCH_SIZE = 64
LEARN_EVERY = 4
START_LEARNING_AFTER = 10_000
REFRESH_TARGET_EVERY = 2_000

START_EPSILON = 1.0
END_EPSILON = 0.05
EXPLORING_SHARE = 0.25


def run_name(switches, decide_every):
    name = "dqn"
    for switch in ("--no-memory", "--no-target", "--no-bricks"):
        if switch in switches:
            name += switch.replace("--", "-")
    if decide_every != 4:
        name += f"-every-{decide_every}"
    return name


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

    decide_every = 4
    if "--decide-every" in sys.argv:
        decide_every = int(sys.argv[sys.argv.index("--decide-every") + 1])
    # Gamma is per decision. Deciding more often means more decisions
    # in the same stretch of play, so gamma moves closer to 1 to keep
    # the player looking the same number of frames ahead.
    player = DQNPlayer(
        see_bricks="--no-bricks" not in sys.argv,
        decide_every=decide_every,
        discount=0.99 ** (decide_every / 4),
    )
    use_target = "--no-target" not in sys.argv
    if "--no-memory" in sys.argv:
        # A memory of one, learned from at every decision: the player
        # only ever learns from the decision it has just made.
        memory = ReplayMemory(1, player.inputs)
        batch_size, learn_every, start_after = 1, 1, 1
    else:
        memory = ReplayMemory(MEMORY_SIZE, player.inputs)
        batch_size = BATCH_SIZE
        learn_every = LEARN_EVERY
        start_after = START_LEARNING_AFTER
    rng = np.random.default_rng(0)

    env = BrickEnv(max_frames=TRAINING_FRAMES)
    log = ProgressLog(
        run_name(sys.argv, decide_every),
        extra="loss",
        report_every=20,
        live_chart="--live" in sys.argv,
    )
    player.save(log.folder / "network-0.pt")

    frames = 0
    decisions = 0
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
            decisions += 1

            new_state = player.state(observation)
            memory.add(state, action, reward, new_state, ended)
            state = new_state

            if (
                memory.count >= start_after
                and decisions % learn_every == 0
            ):
                batch = memory.sample(batch_size, rng)
                losses.append(player.learn(batch, use_target))
            if decisions % REFRESH_TARGET_EVERY == 0:
                player.refresh_target()

        average_loss = float(np.mean(losses)) if losses else 0.0
        log.game_over(info, extra=round(average_loss, 5))

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
