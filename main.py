# Neural Network based self driving car AI
# Version: 0.1
# Author: Guillermo Ochoa
# TODO:
# -Car Distance Sensors
# -Checkpoints and fitness score
# -NN: Finish coding
# -NN: Interfacing car with NN
# -NN: Generation label
# -NN: Network weights and biases saving as json
# -Run Multiple cars at once
# -Able to run training mode or main mode

import pygame as pg
from pygame.locals import *
import sys
import os

WIDTH = 1200
HEIGHT = 900

class Checkpoint(pg.sprite.Sprite):
    def __init__(self, init_xy1, init_xy2, goal=False):
        pg.sprite.Sprite.__init__(self)
        

class Track(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('track.png')
        self.rect = self.image.get_rect()
        self.mask = pg.mask.from_surface(self.image)


class Car(pg.sprite.Sprite):
    def __init__(self, init_x, init_y):
        pg.sprite.Sprite.__init__(self)
        self.pic = pg.image.load('car.png')
        self.image = self.pic
        self.rect = self.image.get_rect(center = (init_x, init_y))
        self.mask = pg.mask.from_surface(self.image)
        self.x = init_x
        self.y = init_y
        self.acceleration = 0.2
        self.desceleration = 0.3
        self.top_speed = 4.0
        self.rotation_speed = 2.0
        self.speed = 0.0
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.accelerate = False
        self.brake = False
        self.turn = 0
        self.angle = 0

    def update(self):
        if (self.turn != 0):
            self.angle -= self.turn*self.rotation_speed
            if(self.angle > 360):self.angle=0
            if(self.angle < 0):self.angle=360

            self.direction.rotate_ip(self.turn*self.rotation_speed)
            self.image, self.rect = self.rotate(self.pic, self.angle)

        if(self.accelerate and self.speed < self.top_speed):
            self.speed += self.acceleration
        elif(self.brake and self.speed > 0.0):
            self.speed -= self.desceleration
        if(not self.brake and not self.accelerate):
            self.speed -= 0.1
        if(self.speed <= 0.0):
            self.speed = 0.0

        self.x += (self.direction * self.speed)[0]
        self.y += (self.direction * self.speed)[1]
        self.rect.center = int(self.x), int(self.y)

    def rotate(self, image, angle):
        rot_image = pg.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=self.rect.center)
        return rot_image,rot_rect
    
    def reset(self, init_x, init_y):
        self.x = init_x
        self.y = init_y
        self.speed = 0
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.angle = 0
        self.image, self.rect = self.rotate(self.pic, self.angle)

def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Handmade Brain")
    clock = pg.time.Clock()

    track = Track()

    car = Car(380, 90)

    while 1:
        # Game Actions
        car.mask = pg.mask.from_surface(car.image)

        # Input Events
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_w:
                    car.accelerate = True
                if event.key == K_a:
                    car.turn = -1
                if event.key == K_d:
                    car.turn = 1
                if event.key == K_s:
                    car.brake = True
            if event.type == KEYUP:
                if event.key == K_w:
                    car.accelerate = False
                if event.key == K_a:
                    car.turn = 0
                if event.key == K_d:
                    car.turn = 0
                if event.key == K_s:
                    car.brake = False

        # Game Events
        if pg.sprite.collide_mask(car, track):
            print("Collision detected!")
            car.reset(380, 90)

        # Updates
        car.update()

        # Drawing
        screen.fill((100, 100, 100))
        screen.blit(track.image, (0, 0))
        screen.blit(car.image, car.rect.topleft)
        pg.display.flip()

        clock.tick(60)


if __name__ == "__main__":
    main()