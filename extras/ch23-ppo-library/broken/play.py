"""Play the game with the keyboard.

Left and right arrows (or A and D) move the paddle. Space starts a game,
P pauses, Escape quits.

Run it with:  uv run play.py
"""

import pygame

from brickgame.game import LEFT, RIGHT, STAY, BrickGame
from brickgame.render import Screen

FPS = 60
LEVEL_BANNER_FRAMES = 90


def keyboard_action():
    keys = pygame.key.get_pressed()
    left = keys[pygame.K_LEFT] or keys[pygame.K_a]
    right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
    if left and not right:
        return LEFT
    if right and not left:
        return RIGHT
    return STAY


def main():
    game = BrickGame()
    screen = Screen()
    started = False
    paused = False
    banner_left = 0
    level_shown = game.level

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    if game.game_over:
                        game.reset()
                        level_shown = game.level
                    started = True
                elif event.key == pygame.K_p and started:
                    paused = not paused

        # A new level holds the game still for a moment so the player can
        # see the new layout before the ball moves.
        if game.level != level_shown:
            level_shown = game.level
            banner_left = LEVEL_BANNER_FRAMES

        if banner_left > 0:
            banner_left -= 1
        elif started and not paused and not game.game_over:
            game.step(keyboard_action())

        if not started:
            message = "Press Space to start"
        elif game.game_over:
            message = f"Game over: level {game.level}, score {game.score}"
            message += "\nSpace to play again"
        elif banner_left > 0:
            message = f"Level {game.level}"
        elif paused:
            message = "Paused"
        else:
            message = None

        screen.draw(game, message)
        screen.wait(FPS)

    screen.close()


if __name__ == "__main__":
    main()
