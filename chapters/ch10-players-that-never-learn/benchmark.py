"""Play 100 games with each player that never learns, and chart them.

Game 1 uses seed 1, game 2 seed 2 and so on, so every player gets the
same serves in the same order.

Run it with:  uv run benchmark.py
"""

from brickgame.env import BrickEnv
from players import PLAYERS, play_game
from training.charts import players_compared
from training.progress import ProgressLog

GAMES = 100


def main():
    env = BrickEnv()
    for name, make_player in PLAYERS.items():
        print(f"\n{name}")
        player = make_player()
        log = ProgressLog(name, report_every=25)
        for game in range(1, GAMES + 1):
            info = play_game(player, env, seed=game)
            log.game_over(info)
        log.close()

    path = players_compared(
        list(PLAYERS),
        "benchmark.png",
        "Thinking ahead breaks the most bricks",
    )
    print(f"\nchart saved to {path}")


if __name__ == "__main__":
    main()
