"""Record examples of where a falling ball comes down.

The chaser plays 20 games. Every few frames while the ball is falling,
the ball's position and speed are noted down. When the ball reaches the
paddle's height, the point where it arrived is written next to each of
those notes. Each row of the file is one example: four numbers that
describe the ball, and the right answer.

Run it with:  uv run record_landings.py
"""

import csv
from pathlib import Path

from brickgame.env import BALL_DX, BALL_DY, BALL_X, BALL_Y, BrickEnv
from brickgame.game import BALL_RADIUS, PADDLE_Y, WIDTH
from players import Chaser

GAMES = 20
NOTE_EVERY = 4
DATA_FILE = Path(__file__).parent / "data" / "landings.csv"
PADDLE_HEIGHT_LINE = PADDLE_Y - BALL_RADIUS


def main():
    DATA_FILE.parent.mkdir(exist_ok=True)
    env = BrickEnv()
    player = Chaser()
    examples = []

    for game in range(1, GAMES + 1):
        observation, info = env.reset(seed=game)
        waiting = []
        finished = False
        while not finished:
            ball = env.game
            if ball.ball_dy > 0 and ball.frames % NOTE_EVERY == 0:
                waiting.append(
                    [
                        observation[BALL_X],
                        observation[BALL_Y],
                        observation[BALL_DX],
                        observation[BALL_DY],
                    ]
                )

            action = player.act(observation)
            result = env.step(action)
            observation, reward, terminated, truncated, info = result
            finished = terminated or truncated

            if not waiting:
                continue
            if ball.ball_y >= PADDLE_HEIGHT_LINE - 0.5:
                # The ball reached the paddle's height, caught or not.
                landing = ball.ball_x / WIDTH * 2 - 1
                for notes in waiting:
                    examples.append([*notes, landing])
                waiting = []
            elif ball.ball_dy < 0:
                # It hit a brick on the way down and went back up.
                waiting = []

    with DATA_FILE.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["ball_x", "ball_y", "ball_dx", "ball_dy", "landing"]
        )
        for row in examples:
            writer.writerow([round(float(value), 5) for value in row])
    print(f"saved {len(examples):,} examples to {DATA_FILE}")


if __name__ == "__main__":
    main()
