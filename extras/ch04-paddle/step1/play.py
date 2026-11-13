import pygame

WIDTH = 600
HEIGHT = 500
FPS = 60

BACKGROUND = (246, 243, 235)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Brick Breaker")
clock = pygame.time.Clock()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(BACKGROUND)
    pygame.display.flip()
    clock.tick(FPS)

pygame.quit()
