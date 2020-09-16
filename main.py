import pygame as pg
from pygame.locals import *
import sys
import os

WIDTH = 1200
HEIGHT = 900


class Track(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('track.png')


class Car(pg.sprite.Sprite):
    def __init__(self, init_x, init_y):
        pg.sprite.Sprite.__init__(self)
        self.pic = pg.image.load('car.png')
        self.image = self.pic
        self.rect = self.image.get_rect(center = (init_x, init_y))
        self.acceleration = 0.2
        self.desceleration = 0.3
        self.rotation_speed = 2.0
        self.speed = 0.0
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.accelerate = False
        self.brake = False
        self.turn_left = False
        self.turn_right = False

    def update(self):
        if (self.turn_left):
            self.direction.rotate_ip(-self.rotation_speed)
        if (self.turn_right):
            self.direction.rotate_ip(self.rotation_speed)
        if(self.turn_left or self.turn_right):
            self.image = pg.transform.rotozoom(self.pic, self.direction.angle_to(pg.math.Vector2(1.0, 0.0)), 0.25)
            self.rect = self.image.get_rect(center=self.rect.center)

        if(self.accelerate and self.speed < 6.0):
            self.speed += self.acceleration
        elif(self.brake and self.speed > 0.0):
            self.speed -= self.desceleration
        if(not self.brake and not self.accelerate):
            self.speed -= 0.1
        if(self.speed <= 0.0):
            self.speed = 0.0
            
        self.rect.center += (self.direction * self.speed)

def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Handmade Brain")
    clock = pg.time.Clock()

    track = Track()
    car = Car(380, 75)

    while 1:
        # Game Events

        # Input Events
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_w:
                    car.accelerate = True
                if event.key == K_a:
                    car.turn_left = True
                if event.key == K_d:
                    car.turn_right = True
                if event.key == K_s:
                    car.brake = True
            if event.type == KEYUP:
                if event.key == K_w:
                    car.accelerate = False
                if event.key == K_a:
                    car.turn_left = False
                if event.key == K_d:
                    car.turn_right = False
                if event.key == K_s:
                    car.brake = False

        # Updates
        clock.tick(60)
        car.update()

        # Drawing
        screen.fill((100, 100, 100))
        screen.blit(track.image, (0, 0))
        screen.blit(car.image, car.rect.center)
        pg.display.flip()


if __name__ == "__main__":
    main()
