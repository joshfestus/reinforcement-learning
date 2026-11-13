import math
import random

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


def serve(paddle_x):
    """Put the ball on the paddle and send it up at a random angle.
    Returns the ball's position and its speed across and down."""
    ball_x = paddle_x + PADDLE_WIDTH / 2
    ball_y = PADDLE_Y - BALL_RADIUS - 1
    angle = math.radians(random.uniform(-45, 45))
    ball_dx = BALL_SPEED * math.sin(angle)
    ball_dy = -BALL_SPEED * math.cos(angle)
    return ball_x, ball_y, ball_dx, ball_dy


pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")
clock = pygame.time.Clock()

paddle_x = (WIDTH - PADDLE_WIDTH) / 2
ball_x, ball_y, ball_dx, ball_dy = serve(paddle_x)

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

    if ball_y - BALL_RADIUS > HEIGHT:
        ball_x, ball_y, ball_dx, ball_dy = serve(paddle_x)

    screen.fill(BACKGROUND)
    paddle = pygame.Rect(paddle_x, PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, INK, paddle, border_radius=4)
    pygame.draw.circle(screen, INK, (ball_x, ball_y), BALL_RADIUS)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
