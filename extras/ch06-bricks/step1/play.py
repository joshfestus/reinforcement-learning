import math
import random
from dataclasses import dataclass

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
ROWS = 5
BRICK_GAP = 4
BRICK_HEIGHT = 20
BRICK_TOP = 60
BRICK_WIDTH = (WIDTH - BRICK_GAP * (COLUMNS + 1)) / COLUMNS

# One color for each row, top row first.
ROW_COLORS = [
    (210, 84, 72),
    (244, 170, 90),
    (246, 222, 120),
    (100, 176, 104),
    (56, 100, 170),
]


@dataclass
class Brick:
    x: float
    y: float
    row: int
    column: int


def build_wall():
    """A list of every brick in the wall, row by row."""
    bricks = []
    for row in range(ROWS):
        for column in range(COLUMNS):
            x = BRICK_GAP + column * (BRICK_WIDTH + BRICK_GAP)
            y = BRICK_TOP + row * (BRICK_HEIGHT + BRICK_GAP)
            bricks.append(Brick(x, y, row, column))
    return bricks


def serve(paddle_x):
    """Put the ball on the paddle and send it up at a random angle.
    Returns the ball's position and its speed across and down."""
    ball_x = paddle_x + PADDLE_WIDTH / 2
    ball_y = PADDLE_Y - BALL_RADIUS - 1
    angle = math.radians(random.uniform(-45, 45))
    ball_dx = BALL_SPEED * math.sin(angle)
    ball_dy = -BALL_SPEED * math.cos(angle)
    return ball_x, ball_y, ball_dx, ball_dy


def meets_paddle(ball_x, ball_y, ball_dy, paddle_x):
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
    right_edge = paddle_x + PADDLE_WIDTH + BALL_RADIUS
    return crossed_top and left_edge <= ball_x <= right_edge


def bounce_off_paddle(ball_x, paddle_x):
    """The ball's new speed across and down after it lands on the paddle.
    The center sends it straight up, the ends at up to 60 degrees."""
    center = paddle_x + PADDLE_WIDTH / 2
    half_width = PADDLE_WIDTH / 2 + BALL_RADIUS
    offset = (ball_x - center) / half_width
    offset = max(-1.0, min(1.0, offset))
    angle = math.radians(offset * 60)
    return BALL_SPEED * math.sin(angle), -BALL_SPEED * math.cos(angle)


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")
clock = pygame.time.Clock()

paddle_x = (WIDTH - PADDLE_WIDTH) / 2
ball_x, ball_y, ball_dx, ball_dy = serve(paddle_x)
bricks = build_wall()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        paddle_x -= PADDLE_SPEED
    if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        paddle_x += PADDLE_SPEED

    if paddle_x < 0:
        paddle_x = 0
    if paddle_x > WIDTH - PADDLE_WIDTH:
        paddle_x = WIDTH - PADDLE_WIDTH

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

    if meets_paddle(ball_x, ball_y, ball_dy, paddle_x):
        ball_dx, ball_dy = bounce_off_paddle(ball_x, paddle_x)
        ball_y = PADDLE_Y - BALL_RADIUS

    if ball_y - BALL_RADIUS > HEIGHT:
        ball_x, ball_y, ball_dx, ball_dy = serve(paddle_x)

    screen.fill(BACKGROUND)
    for brick in bricks:
        box = pygame.Rect(brick.x, brick.y, BRICK_WIDTH, BRICK_HEIGHT)
        color = ROW_COLORS[brick.row]
        pygame.draw.rect(screen, color, box, border_radius=3)
        pygame.draw.rect(screen, INK, box, 1, border_radius=3)

    paddle = pygame.Rect(paddle_x, PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, INK, paddle, border_radius=4)
    pygame.draw.circle(screen, INK, (ball_x, ball_y), BALL_RADIUS)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
