import pygame, numpy, sys
from pygame.locals import *

WIDTH = 800
HEIGHT = 600

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Handmade Brain")
    clock = pygame.time.Clock()

    while 1:
        clock.tick(60)
        pygame.display.update()
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
                if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                    pygame.quit()
                    sys.exit()

        pygame.display.flip()

if __name__ == "__main__":
    main()