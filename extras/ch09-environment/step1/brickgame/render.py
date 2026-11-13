"""Draws the game with pygame.

The screen reads the game and never changes it. It can draw into a window
or, with window=False, into an image in memory that can be saved as a
picture without opening anything.
"""

import pygame

from brickgame.game import (
    BALL_RADIUS,
    BRICK_HEIGHT,
    BRICK_WIDTH,
    HEIGHT,
    PADDLE_HEIGHT,
    PADDLE_Y,
    WIDTH,
)

BACKGROUND = (246, 243, 235)
INK = (40, 40, 48)

# Listed from darkest to lightest. Each color is a different shade of
# gray when the book prints the screen in black and white.
BRICK_COLORS = {
    "P": (96, 56, 116),
    "B": (56, 100, 170),
    "R": (210, 84, 72),
    "G": (100, 176, 104),
    "O": (244, 170, 90),
    "Y": (246, 222, 120),
}


class Screen:
    def __init__(self, window=True, title="Brick Breaker"):
        self.window = window
        pygame.font.init()
        if window:
            pygame.display.init()
            self.surface = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption(title)
        else:
            self.surface = pygame.Surface((WIDTH, HEIGHT))
        self.font = pygame.font.Font(None, 28)
        self.big_font = pygame.font.Font(None, 40)
        self.clock = pygame.time.Clock()

    def draw(self, game, message=None):
        self.surface.fill(BACKGROUND)

        for brick in game.bricks:
            box = pygame.Rect(brick.x, brick.y, BRICK_WIDTH, BRICK_HEIGHT)
            color = BRICK_COLORS[brick.letter]
            pygame.draw.rect(self.surface, color, box, border_radius=3)
            pygame.draw.rect(self.surface, INK, box, 1, border_radius=3)

        paddle = pygame.Rect(
            game.paddle_x, PADDLE_Y, game.paddle_width, PADDLE_HEIGHT
        )
        pygame.draw.rect(self.surface, INK, paddle, border_radius=4)
        ball_center = (game.ball_x, game.ball_y)
        pygame.draw.circle(self.surface, INK, ball_center, BALL_RADIUS)

        status = (
            f"Level {game.level}     Score {game.score}     "
            f"Lives {game.lives}"
        )
        self.write(status, self.font, (WIDTH / 2, 28))

        if message:
            lines = message.split("\n")
            top = HEIGHT * 0.6 - (len(lines) - 1) * 20
            for number, line in enumerate(lines):
                y = top + number * 40
                self.write(line, self.big_font, (WIDTH / 2, y))

        if self.window:
            pygame.display.flip()

    def write(self, text, font, center):
        image = font.render(text, True, INK)
        self.surface.blit(image, image.get_rect(center=center))

    def save(self, path):
        pygame.image.save(self.surface, path)

    def wait(self, fps):
        """Pause just long enough to show `fps` frames a second."""
        self.clock.tick(fps)

    def close(self):
        pygame.quit()
