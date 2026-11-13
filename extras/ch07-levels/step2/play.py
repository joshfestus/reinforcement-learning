import math
import random
from dataclasses import dataclass
from pathlib import Path

import pygame

WIDTH = 600
HEIGHT = 500
FPS = 60

BACKGROUND = (246, 243, 235)
INK = (40, 40, 48)

PADDLE_WIDTH = 80
PADDLE_HEIGHT = 12
PADDLE_Y = HEIGHT - 40
PADDLE_SPEED = 5

BALL_RADIUS = 6
BALL_SPEED = 6

COLUMNS = 8
BRICK_GAP = 4
BRICK_HEIGHT = 20
BRICK_TOP = 60
BRICK_WIDTH = (WIDTH - BRICK_GAP * (COLUMNS + 1)) / COLUMNS

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
LEVELS_FOLDER = Path(__file__).parent / "levels"

START_LIVES = 3

# Each level makes the ball faster and the paddle narrower. The ball is
# capped at 16 pixels a frame: any faster and it could pass a brick
# without ever overlapping it.
SPEEDUP_PER_LEVEL = 0.25
MAX_BALL_SPEED = 16
SHRINK_PER_LEVEL = 15
MIN_PADDLE_WIDTH = 30


@dataclass
class Brick:
    x: float
    y: float
    row: int
    column: int
    letter: str


def read_layout(path):
    """Return (row, column, letter) for every brick in a level file."""
    rows = [line.strip() for line in path.read_text().splitlines()]
    rows = [row for row in rows if row]

    bricks = []
    for row_number, row in enumerate(rows):
        for column, letter in enumerate(row):
            if letter == ".":
                continue
            bricks.append((row_number, column, letter))
    return bricks


def level_layout(level):
    """The layout for a level number, counting from 1."""
    files = sorted(LEVELS_FOLDER.glob("*.txt"))
    return read_layout(files[(level - 1) % len(files)])


def build_wall(level):
    """A list of every brick in the level's wall."""
    bricks = []
    for row, column, letter in level_layout(level):
        x = BRICK_GAP + column * (BRICK_WIDTH + BRICK_GAP)
        y = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
        bricks.append(Brick(x, y, row, column, letter))
    return bricks


def level_settings(level):
    """The ball's speed and the paddle's width on a level."""
    faster = 1 + SPEEDUP_PER_LEVEL * (level - 1)
    ball_speed = min(MAX_BALL_SPEED, BALL_SPEED * faster)
    narrower = SHRINK_PER_LEVEL * (level - 1)
    paddle_width = max(MIN_PADDLE_WIDTH, PADDLE_WIDTH - narrower)
    return ball_speed, paddle_width


def serve(paddle_x, paddle_width, ball_speed):
    """Put the ball on the paddle and send it up at a random angle.
    Returns the ball's position and its speed across and down."""
    ball_x = paddle_x + paddle_width / 2
    ball_y = PADDLE_Y - BALL_RADIUS - 1
    angle = math.radians(random.uniform(-45, 45))
    ball_dx = ball_speed * math.sin(angle)
    ball_dy = -ball_speed * math.cos(angle)
    return ball_x, ball_y, ball_dx, ball_dy


def meets_paddle(ball_x, ball_y, ball_dy, paddle_x, paddle_width):
    """True if the ball reached the paddle during the last frame."""
    if ball_dy < 0:
        return False
    # Compare where the ball's bottom edge is now with where it was a
    # frame ago, so that a fast ball cannot jump over the paddle.
    bottom_now = ball_y + BALL_RADIUS
    bottom_before = bottom_now - ball_dy
    crossed_top = (
        bottom_now >= PADDLE_Y
        and bottom_before <= PADDLE_Y + PADDLE_HEIGHT
    )
    left_edge = paddle_x - BALL_RADIUS
    right_edge = paddle_x + paddle_width + BALL_RADIUS
    return crossed_top and left_edge <= ball_x <= right_edge


def bounce_off_paddle(ball_x, paddle_x, paddle_width, ball_speed):
    """The ball's new speed across and down after it lands on the paddle.
    The center sends it straight up, the ends at up to 60 degrees."""
    center = paddle_x + paddle_width / 2
    half_width = paddle_width / 2 + BALL_RADIUS
    offset = (ball_x - center) / half_width
    offset = max(-1.0, min(1.0, offset))
    angle = math.radians(offset * 60)
    return ball_speed * math.sin(angle), -ball_speed * math.cos(angle)


def break_brick(bricks, ball_x, ball_y, ball_dx, ball_dy):
    """Remove the first brick the ball overlaps and bounce off it.
    Returns whether a brick broke, and the ball's new speeds."""
    for brick in bricks:
        center_x = brick.x + BRICK_WIDTH / 2
        center_y = brick.y + BRICK_HEIGHT / 2
        dx = ball_x - center_x
        dy = ball_y - center_y
        overlap_x = BRICK_WIDTH / 2 + BALL_RADIUS - abs(dx)
        overlap_y = BRICK_HEIGHT / 2 + BALL_RADIUS - abs(dy)
        if overlap_x <= 0 or overlap_y <= 0:
            continue

        # The ball has pushed less far into the side it came through.
        if overlap_x < overlap_y:
            ball_dx = abs(ball_dx)
            if dx < 0:
                ball_dx = -ball_dx
        else:
            ball_dy = abs(ball_dy)
            if dy < 0:
                ball_dy = -ball_dy
        bricks.remove(brick)
        return True, ball_dx, ball_dy
    return False, ball_dx, ball_dy


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 28)
big_font = pygame.font.Font(None, 40)

level = 1
score = 0
lives = START_LIVES
ball_speed, paddle_width = level_settings(level)
paddle_x = (WIDTH - paddle_width) / 2
bricks = build_wall(level)
ball_x, ball_y, ball_dx, ball_dy = serve(
    paddle_x, paddle_width, ball_speed
)
game_over = False

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        pressed = event.type == pygame.KEYDOWN
        if pressed and event.key == pygame.K_SPACE and game_over:
            level = 1
            score = 0
            lives = START_LIVES
            ball_speed, paddle_width = level_settings(level)
            paddle_x = (WIDTH - paddle_width) / 2
            bricks = build_wall(level)
            ball_x, ball_y, ball_dx, ball_dy = serve(
                paddle_x, paddle_width, ball_speed
            )
            game_over = False

    if not game_over:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            paddle_x -= PADDLE_SPEED
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            paddle_x += PADDLE_SPEED
        paddle_x = max(0, min(WIDTH - paddle_width, paddle_x))

        ball_x += ball_dx
        ball_y += ball_dy

        if ball_x < BALL_RADIUS:
            ball_x = BALL_RADIUS
            ball_dx = abs(ball_dx)
        if ball_x > WIDTH - BALL_RADIUS:
            ball_x = WIDTH - BALL_RADIUS
            ball_dx = -abs(ball_dx)
        if ball_y < BALL_RADIUS:
            ball_y = BALL_RADIUS
            ball_dy = abs(ball_dy)

        if meets_paddle(ball_x, ball_y, ball_dy, paddle_x, paddle_width):
            ball_dx, ball_dy = bounce_off_paddle(
                ball_x, paddle_x, paddle_width, ball_speed
            )
            ball_y = PADDLE_Y - BALL_RADIUS

        hit, ball_dx, ball_dy = break_brick(
            bricks, ball_x, ball_y, ball_dx, ball_dy
        )
        if hit:
            score += 1
            if not bricks:
                level += 1
                ball_speed, paddle_width = level_settings(level)
                paddle_x = (WIDTH - paddle_width) / 2
                bricks = build_wall(level)
                ball_x, ball_y, ball_dx, ball_dy = serve(
                    paddle_x, paddle_width, ball_speed
                )

        if ball_y - BALL_RADIUS > HEIGHT:
            lives -= 1
            if lives == 0:
                game_over = True
            else:
                ball_x, ball_y, ball_dx, ball_dy = serve(
                    paddle_x, paddle_width, ball_speed
                )

    screen.fill(BACKGROUND)
    for brick in bricks:
        box = pygame.Rect(brick.x, brick.y, BRICK_WIDTH, BRICK_HEIGHT)
        color = BRICK_COLORS[brick.letter]
        pygame.draw.rect(screen, color, box, border_radius=3)
        pygame.draw.rect(screen, INK, box, 1, border_radius=3)

    paddle = pygame.Rect(paddle_x, PADDLE_Y, paddle_width, PADDLE_HEIGHT)
    pygame.draw.rect(screen, INK, paddle, border_radius=4)
    if not game_over:
        pygame.draw.circle(screen, INK, (ball_x, ball_y), BALL_RADIUS)

    status = f"Level {level}     Score {score}     Lives {lives}"
    text = font.render(status, True, INK)
    screen.blit(text, text.get_rect(center=(WIDTH / 2, 28)))

    if game_over:
        message = f"Game over: level {level}, score {score}"
        for line, y in [(message, 300), ("Space to play again", 340)]:
            text = big_font.render(line, True, INK)
            screen.blit(text, text.get_rect(center=(WIDTH / 2, y)))

    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
