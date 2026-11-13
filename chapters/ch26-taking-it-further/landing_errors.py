"""Find out why good guesses make a poor player.

Sorts the network's guesses on unseen examples by how steeply the ball
was falling, then counts how steeply the balls fall in the games the
network steers itself. Last, the same steering is given exact landing
points instead of the network's guesses.

Run it with:  uv run landing_errors.py   (after train_landing_net.py)
"""

import math

import numpy as np

from brickgame.env import BALL_DX, BALL_DY, BrickEnv
from brickgame.game import BALL_SPEED, HEIGHT, WIDTH
from players import play_game
from players.landing_net import LandingNetwork
from players.landing_player import LandingPlayer
from players.predictor import landing_point
from record_landings import DATA_FILE
from training.progress import RUNS_FOLDER

GAMES = 20
STEEPNESS_BANDS = [(0, 15), (15, 30), (30, 45), (45, 90)]


def steepness(ball_dx, ball_dy):
    """Degrees away from falling straight down."""
    return math.degrees(math.atan2(abs(ball_dx), ball_dy))


def share_by_steepness(angles, misses=None):
    """Print the share of angles in each band, with the average miss."""
    angles = np.array(angles)
    for low, high in STEEPNESS_BANDS:
        inside = (angles >= low) & (angles < high)
        line = f"  {low:2} to {high:2} degrees: {inside.mean():6.1%}"
        if misses is not None:
            line += f"   off by {misses[inside].mean():5.1f} pixels"
        print(line)


class WatchedPlayer(LandingPlayer):
    """The landing player, noting the steepness of every falling ball
    it is asked about."""

    def __init__(self, network):
        super().__init__(network)
        self.angles = []

    def act(self, observation):
        if observation[BALL_DY] > 0:
            angle = steepness(observation[BALL_DX], observation[BALL_DY])
            self.angles.append(angle)
        return super().act(observation)


class ExactLandings:
    """Gives exact landing points, worked out as the predictor does."""

    def predict(self, rows):
        guesses = []
        for ball_x, ball_y, ball_dx, ball_dy in rows:
            x = (ball_x + 1) / 2 * WIDTH
            y = (ball_y + 1) / 2 * HEIGHT
            landing = landing_point(
                x, y, ball_dx * BALL_SPEED, ball_dy * BALL_SPEED
            )
            guesses.append(landing / WIDTH * 2 - 1)
        return np.array(guesses)


def main():
    data = np.loadtxt(
        DATA_FILE, delimiter=",", skiprows=1, dtype=np.float32
    )
    unseen = data[int(len(data) * 0.8) :]
    network = LandingNetwork()
    network.load(RUNS_FOLDER / "landing-numpy" / "network.npz")

    angles = [steepness(row[2], row[3]) for row in unseen]
    misses = np.abs(network.predict(unseen[:, :4]) - unseen[:, 4])
    print("the recorded falls it never trained on:")
    share_by_steepness(angles, misses * WIDTH / 2)

    env = BrickEnv()
    player = WatchedPlayer(network)
    scores = [
        play_game(player, env, seed=g)["score"] for g in range(1, 21)
    ]
    print(f"\nthe falls in {GAMES} games it steered itself:")
    share_by_steepness(player.angles)
    print(f"  {np.mean(scores):.1f} bricks a game")

    exact = LandingPlayer(ExactLandings())
    scores = [
        play_game(exact, env, seed=g)["score"] for g in range(1, 21)
    ]
    print(
        f"\nthe same steering with exact landing points: "
        f"{np.mean(scores):.1f} bricks a game"
    )


if __name__ == "__main__":
    main()
