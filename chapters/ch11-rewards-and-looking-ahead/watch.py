"""Watch a player in a window.

    uv run watch.py chaser
    uv run watch.py predictor random

Several names play one after another. Keys while watching: + and - change
the speed, N skips to the next player, Escape quits.
"""

import sys

import pygame

from brickgame.env import BrickEnv
from brickgame.render import Screen
from players import PLAYERS


def watch(player, env, screen, fps):
    """Play one game on screen. Returns the speed and whether to go on."""
    observation, info = env.reset()
    finished = False
    while not finished:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return fps, False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return fps, False
                if event.key == pygame.K_n:
                    return fps, True
                if event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(fps * 2, 960)
                if event.key == pygame.K_MINUS:
                    fps = max(fps // 2, 15)

        action = player.act(observation)
        result = env.step(action)
        observation, reward, terminated, truncated, info = result
        finished = terminated or truncated

        screen.draw(env.game, caption=f"{player.name}   {fps / 60:g}x")
        screen.wait(fps)

    ending = info["ending"]
    message = f"{player.name}: {ending}\nlevel {info['level']}, "
    message += f"{info['score']} bricks"
    screen.draw(env.game, message=message)
    pygame.time.wait(2500)
    return fps, True


def main():
    names = sys.argv[1:]
    unknown = [name for name in names if name not in PLAYERS]
    if not names or unknown:
        print(__doc__)
        print("players:", ", ".join(PLAYERS))
        sys.exit(1)

    env = BrickEnv()
    screen = Screen(title="Brick Breaker: watching")
    fps = 60
    for name in names:
        fps, keep_going = watch(PLAYERS[name](), env, screen, fps)
        if not keep_going:
            break
    screen.close()


if __name__ == "__main__":
    main()
