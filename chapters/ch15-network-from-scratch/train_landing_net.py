"""Teach a network where the ball will land, from recorded examples.

This is supervised learning: every example comes with the right answer.
Run record_landings.py first.

Run it with:  uv run train_landing_net.py
"""

import numpy as np

from brickgame.env import BrickEnv
from brickgame.game import WIDTH
from players import play_game
from players.landing_net import LandingNetwork
from players.landing_player import LandingPlayer
from record_landings import DATA_FILE
from training.charts import landing_charts
from training.progress import RUNS_FOLDER

ROUNDS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.02
TEST_GAMES = 20


def pixels(scaled_error):
    """Turn an error in the scaled numbers back into pixels."""
    return scaled_error * WIDTH / 2


def main():
    network = LandingNetwork()
    run_name, file_name = "landing-numpy", "network.npz"

    data = np.loadtxt(
        DATA_FILE, delimiter=",", skiprows=1, dtype=np.float32
    )
    inputs, answers = data[:, :4], data[:, 4]

    # The last fifth of the examples is kept back. The network never
    # trains on it, so it shows how well the network does on falls it
    # has not seen.
    split = int(len(data) * 0.8)
    train_inputs, train_answers = inputs[:split], answers[:split]
    check_inputs, check_answers = inputs[split:], answers[split:]

    rng = np.random.default_rng(0)
    misses = []
    for round_number in range(1, ROUNDS + 1):
        order = rng.permutation(split)
        for start in range(0, split, BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            network.train_step(
                train_inputs[batch], train_answers[batch], LEARNING_RATE
            )

        guesses = network.predict(check_inputs)
        miss = pixels(np.mean(np.abs(guesses - check_answers)))
        misses.append(miss)
        print(
            f"round {round_number:2}   off by "
            f"{miss:5.1f} pixels on average, on falls it has not seen"
        )

    folder = RUNS_FOLDER / run_name
    folder.mkdir(parents=True, exist_ok=True)
    network.save(folder / file_name)

    guesses = network.predict(check_inputs)
    chart = landing_charts(
        misses, pixels(check_answers), pixels(guesses), f"{run_name}.png"
    )
    print(f"chart saved to {chart}")

    env = BrickEnv()
    player = LandingPlayer(network)
    scores = []
    for game in range(1, TEST_GAMES + 1):
        scores.append(play_game(player, env, seed=game)["score"])
    print(
        f"\n{TEST_GAMES} games steered by the network: "
        f"{sum(scores) / len(scores):.1f} bricks on average, "
        f"best {max(scores)}"
    )


if __name__ == "__main__":
    main()
