"""Measure what holds the Q-table back.

The chaser and the predictor play the benchmark's 100 games again, once
deciding every frame as before and once keeping each action for 4
frames, as the table does. The chaser also plays seeing only the
table's buckets. The best saved table plays the same games.

Run it with:  uv run table_limits.py
"""

from brickgame.env import BrickEnv
from brickgame.game import LEFT, RIGHT, STAY
from players import Chaser, Predictor, play_game
from players.qtable import QTablePlayer, situation
from training.progress import RUNS_FOLDER

GAMES = 100


class EveryFour:
    """Any player, made to keep each action for 4 frames."""

    def __init__(self, player):
        self.player = player
        self.action = STAY
        self.frames_left = 0

    def act(self, observation):
        if self.frames_left == 0:
            self.action = self.player.act(observation)
            self.frames_left = 4
        self.frames_left -= 1
        return self.action


class BucketChaser:
    """The chaser's rule, seeing only which bucket the ball is in."""

    def act(self, observation):
        offset = situation(observation)[0]
        if offset <= 5:
            return LEFT
        if offset >= 8:
            return RIGHT
        return STAY


def average_score(player):
    env = BrickEnv()
    total = 0
    for game in range(1, GAMES + 1):
        total += play_game(player, env, seed=game)["score"]
    return total / GAMES


def main():
    table = QTablePlayer(epsilon=0)
    table.load(RUNS_FOLDER / "qtable" / "best.json")
    players = {
        "chaser, every frame": Chaser(),
        "chaser, buckets only, every frame": BucketChaser(),
        "chaser, every 4 frames": EveryFour(Chaser()),
        "chaser, buckets only, every 4 frames": EveryFour(BucketChaser()),
        "predictor, every frame": Predictor(),
        "predictor, every 4 frames": EveryFour(Predictor()),
        "best Q-table (every 4 frames)": table,
    }
    for name, player in players.items():
        print(f"{name:38} {average_score(player):6.1f} bricks a game")


if __name__ == "__main__":
    main()
