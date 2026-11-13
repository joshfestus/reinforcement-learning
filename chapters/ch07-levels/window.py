import pygame

width = 600
height = 500
background = (246, 243, 235)

pygame.init()
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Brick Breaker")

screen.fill(background)
pygame.display.flip()
pygame.event.pump()

pygame.time.wait(5000)
pygame.quit()
