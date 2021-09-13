# Neural Network based self driving car AI
# Version: 0.5
# Author: Guillermo Ochoa
# CHALLENGES:
# -Save best Model
# -Load Model
# -Improve Sensors (convert to distance maybe)
# -Change Car Movement (Acceleration, Braking)
# -Try other Tracks

import pygame as pg
import neuralnet as nn
import numpy as np
from pygame.locals import *
import sys

WIDTH = 1200
HEIGHT = 900

GEN_SIZE = 50
CAR_X = 500
CAR_Y = 150
LAYER_NEURONS = 12
WM_F = 0.5
BM_F = 0.1

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

class Anti_Checkpoints(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('anti_checkpoints.png')
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
        self.acceleration = 0.8
        self.desceleration = 0.4
        self.breaking_force = 0.15
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
        self.sfront.pic = pg.transform.scale(self.sfront.pic, (260, 2))
        self.sleft = Sensor(0, 0)
        self.sleft.pic = pg.transform.scale(self.sleft.pic, (150, 2))
        self.sright = Sensor(0, 0)
        self.sright.pic = pg.transform.scale(self.sright.pic, (150, 2))
        # Pointing System
        self.score = 0
        self.score_colliding = False
        self.anti_colliding = False
        # Inputs
        self.sfront.detection = False
        self.sleft.detection = False
        self.sright.detection = False
        # Outputs
        self.turn = 0
        self.turn_left = 0
        self.turn_right = 0
        self.accelerate = False
        self.brake = False
        # Neural Net Initializers
        i_weight = 0.01
        i_bias = 0.001
        self.dense1 = nn.Layer_Dense(3, LAYER_NEURONS, i_weight, i_bias)
        self.activation1 = nn.Activation_ReLU()
        self.dense2 = nn.Layer_Dense(LAYER_NEURONS, 4, i_weight, i_bias)
        self.activation2 = nn.Activation_ReLU()

    def update(self):
        if not self.crashed:
            # Neural Net
            n_input = np.array(
                [self.sfront.detection, self.sleft.detection, self.sright.detection], dtype=np.float32)
            self.dense1.forward(n_input)
            self.activation1.forward(self.dense1.output)
            self.dense2.forward(self.activation1.output)
            self.activation2.forward(self.dense2.output)
            # Outputs
            self.accelerate = self.activation2.output[0][0]
            self.brake = self.activation2.output[0][1]
            self.turn_left = self.activation2.output[0][2]
            self.turn_right = self.activation2.output[0][3]
            if self.turn_left:
                self.turn = -1
            if self.turn_right:
                self.turn = 1
            if not self.turn_left and not self.turn_right:
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
                self.speed -= self.breaking_force
            if(not self.brake and not self.accelerate):
                self.speed -= self.desceleration
            if(self.speed <= 0.0):
                self.speed = 0.0

            self.x += (self.direction * self.speed)[0]
            self.y += (self.direction * self.speed)[1]
            self.rect.center = int(self.x), int(self.y)

            # Sensors
            self.sfront.update(self.rect.centerx + int((self.direction*160)[0]),
                               self.rect.centery + int((self.direction*160)[1]), self.angle)
            self.sleft.update(self.rect.centerx + int((self.vl*90)[0]),
                              self.rect.centery + int((self.vl*90)[1]), self.angle+20)
            self.sright.update(self.rect.centerx + int((self.vr*90)[0]),
                               self.rect.centery + int((self.vr*90)[1]), self.angle-20)

    def rotate(self, image, angle):
        rot_image = pg.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=self.rect.center)
        return rot_image, rot_rect

    def angle_reset(self):
        # Fix for visual glitches after multiple rotations
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


def genCars(cars_group, w1=np.arange(3, LAYER_NEURONS), b1=np.arange(1, LAYER_NEURONS), w2=np.arange(3, LAYER_NEURONS), b2=np.arange(1, LAYER_NEURONS), evolve=False):
    if evolve:
        # Evolve after all cars crash or E is pressed, best car is clones once, rest of cars are mutated from best car
        for i in range(GEN_SIZE):
            if i != GEN_SIZE-1:
                c = Car(CAR_X, CAR_Y)
                c.dense1.inherit_and_evolve_WB(w1, b1, wmf=WM_F*i, bmf=BM_F*i)
                c.dense2.inherit_and_evolve_WB(w2, b2)
                cars_group.add(c)
            else:
                c = Car(CAR_X, CAR_Y)
                c.dense1.inherit_WB(w1, b1)
                c.dense2.inherit_WB(w2, b2)
                cars_group.add(c)
    else:
        # Fresh Start Cars
        for i in range(GEN_SIZE):
            cars_group.add(Car(CAR_X, CAR_Y))


def main():
    pg.init()
    screen = pg.display.set_mode((WIDTH, HEIGHT))
    pg.display.set_caption("Handmade Brain")
    clock = pg.time.Clock()
    check_time = pg.time.get_ticks()

    # Content Setup
    track = Track()
    checkpoints = Checkpoints()
    anti_checkpoints = Anti_Checkpoints()
    draw_checkpoints = False
    pressing_c = False
    draw_sensors = True
    pressing_s = False

    # Score Setup
    font = pg.font.Font(pg.font.get_default_font(), 20)
    highest_score = 0

    # Cars
    cars_crashed = 0
    cars = pg.sprite.Group()
    genCars(cars)

    # Evolution
    delta_time = 1500 #Time between checkpoints before force evolve
    generation = 1
    hs_w1 = np.arange(3, LAYER_NEURONS)
    hs_b1 = np.arange(1, LAYER_NEURONS)
    hs_w2 = np.arange(3, LAYER_NEURONS)
    hs_b2 = np.arange(1, LAYER_NEURONS)

    while 1:

        # Input Events
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pg.quit()
                sys.exit()
            if event.type == KEYDOWN:
                if event.key == K_r:
                    # Restarts Evolution
                    cars.empty()
                    highest_score = 0
                    cars_crashed = 0
                    generation = 0
                    hs_w1 = np.zeros_like(hs_w1)
                    hs_b1 = np.zeros_like(hs_w1)
                    hs_w2 = np.zeros_like(hs_w1)
                    hs_b2 = np.zeros_like(hs_w1)
                    genCars(cars)
                if event.key == K_e:
                    # Force Evolve
                    cars.empty()
                    highest_score = 0
                    cars_crashed = 0
                    generation += 1
                    genCars(cars, w1=hs_w1, b1=hs_b1, w2=hs_w2, b2=hs_b2, evolve=True)
                if event.key == K_c and not pressing_c:
                    if draw_checkpoints:
                        draw_checkpoints = False
                    else:
                        draw_checkpoints = True
                    pressing_c = True
                if event.key == K_s and not pressing_s:
                    if draw_sensors:
                        draw_sensors = False
                    else:
                        draw_sensors = True
                    pressing_s = True
            if event.type == KEYUP:
                if event.key == K_c:
                    pressing_c = False
                if event.key == K_s:
                    pressing_s = False

        # Game Events
        for i in cars:
            if not i.crashed:
                if i.score < highest_score-20:
                    # Crash car if not scoring
                    i.crashed = True
                    i.accelerate = False
                    i.turn = False
                    i.brake = False
                    cars_crashed += 1
                if pg.sprite.collide_mask(i, track):
                    # Crash with Track
                    i.crashed = True
                    i.accelerate = False
                    i.turn = False
                    i.brake = False
                    cars_crashed += 1
                if pg.sprite.collide_mask(i, checkpoints):
                    # Score on checkpoint
                    if not i.score_colliding:
                        check_time = pg.time.get_ticks()
                        i.score += 10
                        i.score_colliding = True
                if pg.sprite.collide_mask(i, anti_checkpoints):
                    # Checks if car is going wrong way
                    if i.score_colliding:
                        i.anti_colliding = True
                    else:
                        i.crashed = True
                        i.accelerate = False
                        i.turn = False
                        i.brake = False
                        cars_crashed += 1
                if not pg.sprite.collide_mask(i, anti_checkpoints) and not pg.sprite.collide_mask(i, checkpoints):
                    # Checks when successfully passing checkpoint
                    if i.score_colliding:
                        i.score_colliding = False
                
                # Sensors
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
                # Gets highest score and saves car's data
                if i.score >= highest_score:
                    try:
                        output = i.activation2.output
                    except:
                        pass
                    highest_score = i.score
                    hs_w1 = i.dense1.weights
                    hs_b1 = i.dense1.biases
                    hs_w2 = i.dense2.weights
                    hs_b2 = i.dense2.biases

        if cars_crashed == GEN_SIZE:
            # Evolve after all cars crash
            cars.empty()
            highest_score = 0
            cars_crashed = 0
            generation += 1
            genCars(cars, w1=hs_w1, b1=hs_b1, w2=hs_w2, b2=hs_b2, evolve=True)
        
        if check_time <= pg.time.get_ticks()-delta_time:
            # Evolve after too much time passed between checkpoints
            check_time = pg.time.get_ticks()
            cars.empty()
            highest_score = 0
            cars_crashed = 0
            generation += 1
            genCars(cars, w1=hs_w1, b1=hs_b1, w2=hs_w2, b2=hs_b2, evolve=True)

        # Drawing
        screen.fill((100, 100, 100))
        if draw_checkpoints:
            screen.blit(checkpoints.image, (0, 0))
            screen.blit(anti_checkpoints.image, (0, 0))
        screen.blit(track.image, (0, 0))
        cars.update()
        cars.draw(screen)
        if draw_sensors:
            for i in cars:
                if not i.crashed:
                    screen.blit(i.sfront.image, i.sfront.rect.topleft)
                    screen.blit(i.sleft.image, i.sleft.rect.topleft)
                    screen.blit(i.sright.image, i.sright.rect.topleft)

        screen.blit(pg.font.Font.render(font, "Highest Score: " +
                                        str(highest_score), True, (255, 255, 255)), (5, 10))
        screen.blit(pg.font.Font.render(font, "Crashed: " +
                                        str(cars_crashed), True, (255, 255, 255)), (250, 10))
        screen.blit(pg.font.Font.render(font, "Generation: " +
                                        str(generation), True, (255, 255, 255)), (400, 10))

        screen.blit(pg.font.Font.render(font, "Time between checkpoints: " + str((pg.time.get_ticks() - check_time)/1000), True, (255, 255, 255)), (600, 10))

        pg.display.flip()

        clock.tick(144)


if __name__ == "__main__":
    main()
