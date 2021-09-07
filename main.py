# Neural Network based self driving car AI
# Version: 0.5
# Author: Guillermo Ochoa
# TODO:
# -NN: Generation label
# -NN: Network weights and biases saving as json
# -Able to run training mode or main mode

import pygame as pg
import neuralnet as nn
import numpy as np
from pygame.locals import *
import sys
# import os

WIDTH = 1200
HEIGHT = 900


class Track(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('track.png')
        self.rect = self.image.get_rect()
        self.mask = pg.mask.from_surface(self.image)


class Checkpoints(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('checkpoints.png')
        self.rect = self.image.get_rect()
        self.mask = pg.mask.from_surface(self.image)


class Car(pg.sprite.Sprite):
    def __init__(self, init_x, init_y):
        pg.sprite.Sprite.__init__(self)
        self.pic = pg.image.load('car.png')
        self.image = self.pic
        self.rect = self.image.get_rect(center=(init_x, init_y))
        self.mask = pg.mask.from_surface(self.image)
        self.x = init_x
        self.y = init_y
        # Movement
        self.acceleration = 0.2
        self.desceleration = 0.3
        self.top_speed = 4.0
        self.rotation_speed = 2.0
        self.speed = 0.0
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.angle = 0
        self.crashed = False
        # Sensors
        self.vl = pg.math.Vector2(1.0, -0.5)
        self.vr = pg.math.Vector2(1.0, 0.5)
        self.sfront = Sensor(0, 0)
        self.sfront.pic = pg.transform.scale(self.sfront.pic, (200, 2))
        self.sleft = Sensor(0, 0)
        self.sright = Sensor(0, 0)
        # Pointing System
        self.score = 0
        self.score_colliding = False
        # Inputs
        self.sfront.detection = False
        self.sleft.detection = False
        self.sright.detection = False
        # Outputs
        self.turn = 0
        self.accelerate = False
        self.brake = False
        # Neural Net
        i_weight = 0.5
        i_bias = 0.1

        self.dense1 = nn.Layer_Dense(3, 64, i_weight, i_bias)
        self.activation1 = nn.Activation_ReLU()
        self.dense2 = nn.Layer_Dense(64, 3, i_weight, i_bias)
        self.activation2 = nn.Activation_Sigmoid()

    def update(self):
        if self.crashed == False:
            # Neural Net
            n_input = np.array([self.sfront.detection, self.sleft.detection, self.sright.detection], dtype=np.float32)
            self.dense1.forward(n_input)
            self.activation1.forward(self.dense1.output)
            self.dense2.forward(self.activation1.output)
            self.activation2.forward(self.dense2.output)
            # print(f"forward: {self.activation2.output[0][0]}, left: {self.activation2.output[0][1]}, right: {self.activation2.output[0][2]}")
            # print(f"forward: {self.dense2.output[0][0]}, left: {self.dense2.output[0][1]}, right: {self.dense2.output[0][2]}")
            self.accelerate = self.activation2.output[0][0]
            if (self.activation2.output[0][1] > self.activation2.output[0][2]):
                self.turn = -1
            else:
                self.turn = 1
            if (self.activation2.output[0][1] == self.activation2.output[0][2]):
                self.turn = 0

            # Car Mechanics
            if (self.turn != 0):
                self.angle -= self.turn*self.rotation_speed
                if(self.angle > 360):
                    self.angle = 0
                if(self.angle < 0):
                    self.angle = 360

                self.direction.rotate_ip(self.turn*self.rotation_speed)
                self.vl.rotate_ip(self.turn*self.rotation_speed)
                self.vr.rotate_ip(self.turn*self.rotation_speed)
                self.image, self.rect = self.rotate(self.pic, self.angle)
                self.mask = pg.mask.from_surface(self.image)

                if self.angle == 0:
                    self.angle_reset()

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
            # Sensors
            self.sfront.update(self.rect.centerx + int((self.direction*130)[0]),
                            self.rect.centery + int((self.direction*130)[1]), self.angle)
            self.sleft.update(self.rect.centerx + int((self.vl*70)[0]),
                            self.rect.centery + int((self.vl*70)[1]), self.angle+20)
            self.sright.update(self.rect.centerx + int((self.vr*70)[0]),
                            self.rect.centery + int((self.vr*70)[1]), self.angle-20)
            

    def rotate(self, image, angle):
        rot_image = pg.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=self.rect.center)
        return rot_image, rot_rect

    def reset(self, init_x, init_y):
        self.x = init_x
        self.y = init_y
        self.speed = 0
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.angle = 0
        self.image, self.rect = self.rotate(self.pic, self.angle)
        self.vl = pg.math.Vector2(1.0, -0.5)
        self.vr = pg.math.Vector2(1.0, 0.5)

    def angle_reset(self):
        self.image = self.pic
        self.rect = self.image.get_rect(center=self.rect.center)
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.vl = pg.math.Vector2(1.0, -0.5)
        self.vr = pg.math.Vector2(1.0, 0.5)
        self.sfront.image = self.sfront.pic
        self.sfront.rect = self.sfront.image.get_rect()
        self.sleft.image = self.sleft.pic
        self.sleft.rect = self.sleft.image.get_rect()
        self.sright.image = self.sright.pic
        self.sright.rect = self.sright.image.get_rect()


class Sensor(pg.sprite.Sprite):
    def __init__(self, init_x, init_y):
        pg.sprite.Sprite.__init__(self)
        self.pic = pg.image.load('sensor.png')
        self.image = self.pic
        self.rect = self.image.get_rect(center=(init_x, init_y))
        self.mask = pg.mask.from_surface(self.image)
        self.detection = False

    def update(self, xpos, ypos, angle):
        self.image, self.rect = self.rotate(self.pic, angle)
        self.rect.center = int(xpos), int(ypos)
        self.mask = pg.mask.from_surface(self.image)
        if (self.detection):
            self.pic.fill((255, 0, 0))
        else:
            self.pic.fill((0, 255, 0))

    def rotate(self, image, angle):
        rot_image = pg.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=self.rect.center)
        return rot_image, rot_rect

def genCars(cars_group):
    for i in range(20):
            cars_group.add(Car(430, 150))

def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Handmade Brain")
    clock = pg.time.Clock()

    # Content Setup
    track = Track()
    checkpoints = Checkpoints()

    # Score Setup
    font = pg.font.Font(pg.font.get_default_font(), 20)
    highest_score = 0

    # Cars
    cars = pg.sprite.Group()
    genCars(cars)
    
    while 1:

        # Input Events
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_r:
                    cars.empty()
                    genCars(cars)
                    

            # for i in cars:
            #     if i.crashed == False:
            #         if event.type == KEYDOWN:
            #             if event.key == K_w:
            #                 i.accelerate = True
            #             if event.key == K_a:
            #                 i.turn = -1
            #             if event.key == K_d:
            #                 i.turn = 1
            #             if event.key == K_s:
            #                 i.brake = True
            #         if event.type == KEYUP:
            #             if event.key == K_w:
            #                 i.accelerate = False
            #             if event.key == K_a:
            #                 i.turn = 0
            #             if event.key == K_d:
            #                 i.turn = 0
            #             if event.key == K_s:
            #                 i.brake = False

        # Game Events
        for i in cars:
            if pg.sprite.collide_mask(i, track):
                # i.reset(380, 90)
                i.crashed = True
                i.accelerate = False
                i.turn = False
                i.brake = False
            if pg.sprite.collide_mask(i, checkpoints):
                if not i.score_colliding:
                    i.score += 10
                    i.score_colliding = True
            else:
                i.score_colliding = False
            if pg.sprite.collide_mask(i.sfront, track):
                i.sfront.detection = True
            else:
                i.sfront.detection = False
            if pg.sprite.collide_mask(i.sleft, track):
                i.sleft.detection = True
            else:
                i.sleft.detection = False
            if pg.sprite.collide_mask(i.sright, track):
                i.sright.detection = True
            else:
                i.sright.detection = False
            if i.score > highest_score:
                highest_score = i.score

        # Drawing
        screen.fill((100, 100, 100))
        screen.blit(checkpoints.image, (0, 0))
        screen.blit(track.image, (0, 0))
        cars.update()
        cars.draw(screen)
        for i in cars:
            if not i.crashed:
                screen.blit(i.sfront.image, i.sfront.rect.topleft)
                screen.blit(i.sleft.image, i.sleft.rect.topleft)
                screen.blit(i.sright.image, i.sright.rect.topleft)

        screen.blit(pg.font.Font.render(font, "Highest Score: " +
                                        str(highest_score), True, (255, 255, 255)), (5, 10))

        pg.display.flip()

        clock.tick(60)


if __name__ == "__main__":
    main()
