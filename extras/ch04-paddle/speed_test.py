import pygame

pygame.init()
screen = pygame.display.set_mode((600, 500))
clock = pygame.time.Clock()

frames = 0
start = pygame.time.get_ticks()
while pygame.time.get_ticks() - start < 2000:
    pygame.event.pump()
    screen.fill((246, 243, 235))
    pygame.display.flip()
    frames += 1
print("Without the clock:", frames, "frames in two seconds")

frames = 0
start = pygame.time.get_ticks()
while pygame.time.get_ticks() - start < 2000:
    pygame.event.pump()
    screen.fill((246, 243, 235))
    pygame.display.flip()
    clock.tick(60)
    frames += 1
print("With clock.tick(60):", frames, "frames in two seconds")

pygame.quit()
