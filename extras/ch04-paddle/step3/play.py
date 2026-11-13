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

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")
clock = pygame.time.Clock()

paddle_x = (WIDTH - PADDLE_WIDTH) / 2

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

    screen.fill(BACKGROUND)
    paddle = pygame.Rect(paddle_x, PADDLE_Y, PADDLE_WIDTH, PADDLE_HEIGHT)
    pygame.draw.rect(screen, INK, paddle, border_radius=4)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
