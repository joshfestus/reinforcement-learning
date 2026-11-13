"""Let a bandit find out which player scores best, while it plays.

The bandit chooses one of the three players for each game and learns
from the score.

Run it with:  uv run train_bandit.py
"""

from brickgame.env import BrickEnv
from players import PLAYERS, play_game
from players.bandit import Bandit
from training.progress import ProgressLog

GAMES = 100
EPSILON = 0.1


def main():
    names = list(PLAYERS)
    players = [PLAYERS[name]() for name in names]
    bandit = Bandit(names, epsilon=EPSILON, seed=0)
    env = BrickEnv()
    log = ProgressLog("bandit", extra="choice", report_every=10)

    for game in range(1, GAMES + 1):
        arm = bandit.choose()
        info = play_game(players[arm], env, seed=game)
        bandit.learn(arm, info["score"])
        log.game_over(info, extra=names[arm])
    log.close()

    print("\nwhat the bandit learned")
    for arm, name in enumerate(names):
        print(
            f"{name:>10}: played {bandit.counts[arm]:3} times, "
            f"estimate {bandit.estimates[arm]:6.1f} bricks"
        )


if __name__ == "__main__":
    main()
