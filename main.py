# Neural Network based self driving car learning AI
# Version: 1.0
# Author: Guillermo Ochoa
# CHALLENGES:
# -Save best Model to drive
# -Load Model from drive

import pygame as pg
import neuralnet as nn
import numpy as np
from pygame.locals import *
import sys

WIDTH = 1200
HEIGHT = 900

TRACK_N = 2
CAR_X = 550
CAR_Y = 190

SENS_LEN = 240
SENS_AM = 8

GEN_SIZE = 40
LAYER_NEURONS = 30
WM_F = 0.005
BM_F = 0.005

class Track(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('track'+str(TRACK_N)+'.png')
        self.rect = self.image.get_rect()
        self.mask = pg.mask.from_surface(self.image)


class Checkpoints(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('checkpoints'+str(TRACK_N)+'.png')
        self.rect = self.image.get_rect()
        self.mask = pg.mask.from_surface(self.image)


class Anti_Checkpoints(pg.sprite.Sprite):
    def __init__(self):
        pg.sprite.Sprite.__init__(self)
        self.image = pg.image.load('anti_checkpoints'+str(TRACK_N)+'.png')
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
        self.acceleration = 0.15
        self.desceleration = 0.04
        self.breaking_force = 0.2
        self.top_speed = 4.0
        self.rotation_speed = 2.0
        self.speed = 0.0
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.angle = 0
        self.crashed = False
        # Sensors
        self.sensors_amount = SENS_AM
        self.sensors_len = SENS_LEN
        self.sfront = pg.sprite.Group()
        self.sleft = pg.sprite.Group()
        self.sright = pg.sprite.Group()

        for f in range(self.sensors_amount):
            self.sfront.add(Sensor(0, 0))
            self.sfront.sprites()[f].pic = pg.transform.scale(self.sfront.sprites()[
                f].pic, (int(self.sensors_len/self.sensors_amount), 2))
            self.sleft.add(Sensor(0, 0))
            self.sleft.sprites()[f].pic = pg.transform.scale(self.sleft.sprites()[
                f].pic, (int(self.sensors_len/self.sensors_amount), 2))
            self.sright.add(Sensor(0, 0))
            self.sright.sprites()[f].pic = pg.transform.scale(self.sright.sprites()[
                f].pic, (int(self.sensors_len/self.sensors_amount), 2))

        self.vl = pg.math.Vector2(0.5, -0.5)
        self.vr = pg.math.Vector2(0.5, 0.5)
        # Pointing System
        self.score = 0
        self.score_colliding = False
        self.anti_colliding = False
        # Inputs
        self.sfront_distance = self.sensors_len + self.sensors_len/self.sensors_amount
        self.sleft_distance = self.sensors_len + self.sensors_len/self.sensors_amount
        self.sright_distance = self.sensors_len + self.sensors_len/self.sensors_amount
        # Outputs
        self.turn = 0
        self.turn_left = 0
        self.turn_right = 0
        self.accelerate = False
        self.brake = False
        # Neural Net Initializers
        i_weight = 0.01
        i_bias = 0.001
        self.dense1 = nn.Layer_Dense(4, LAYER_NEURONS, i_weight, i_bias)
        self.activation1 = nn.Activation_ReLU()
        self.dense2 = nn.Layer_Dense(LAYER_NEURONS, 4, i_weight, i_bias)
        self.activation2 = nn.Activation_ReLU()

    def update(self):
        if not self.crashed:
            # Neural Net
            n_input = np.array(
                [self.speed, self.sfront_distance, self.sleft_distance, self.sright_distance], dtype=np.float32)
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

            for s in range(self.sensors_amount):
                self.sfront.sprites()[s].update(
                    self.x + (self.direction*self.sensors_len *
                              (s+1)/(self.sensors_amount))[0],
                    self.y + (self.direction*self.sensors_len *
                              (s+1)/(self.sensors_amount))[1],
                    self.angle)

                self.sleft.sprites()[s].update(
                    self.x + (self.vl*self.sensors_len *
                              (s+1)/(self.sensors_amount))[0],
                    self.y + (self.vl*self.sensors_len *
                              (s+1)/(self.sensors_amount))[1],
                    self.angle+45)

                self.sright.sprites()[s].update(
                    self.x + (self.vr*self.sensors_len *
                              (s+1)/(self.sensors_amount))[0],
                    self.y + (self.vr*self.sensors_len *
                              (s+1)/(self.sensors_amount))[1],
                    self.angle-45)

    def rotate(self, image, angle):
        rot_image = pg.transform.rotate(image, angle)
        rot_rect = rot_image.get_rect(center=self.rect.center)
        return rot_image, rot_rect

    def angle_reset(self):
        # Fix for visual glitches after multiple rotations
        self.image = self.pic
        self.rect = self.image.get_rect(center=self.rect.center)
        self.direction = pg.math.Vector2(1.0, 0.0)
        self.vl = pg.math.Vector2(0.5, -0.5)
        self.vr = pg.math.Vector2(0.5, 0.5)
        for s in range(self.sensors_amount):
            self.sfront.sprites()[s].image = self.sfront.sprites()[s].pic
            self.sfront.sprites()[s].rect = self.sfront.sprites()[
                s].image.get_rect()
            self.sleft.sprites()[s].image = self.sleft.sprites()[s].pic
            self.sleft.sprites()[s].rect = self.sleft.sprites()[
                s].image.get_rect()
            self.sright.sprites()[s].image = self.sright.sprites()[s].pic
            self.sright.sprites()[s].rect = self.sright.sprites()[
                s].image.get_rect()


class Sensor(pg.sprite.Sprite):
    def __init__(self, init_x, init_y):
        pg.sprite.Sprite.__init__(self)
        self.pic = pg.image.load('sensor.png')
        self.image = self.pic
        self.rect = self.image.get_rect(center=(init_x, init_y))
        self.mask = pg.mask.from_surface(self.image)
        self.detection = True

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
        # Evolve after all cars crash or E is pressed, best car is cloned once, rest of cars are mutated from best car
        for i in range(GEN_SIZE):
            c = Car(CAR_X, CAR_Y)
            c.dense1.inherit_and_evolve_WB(w1, b1, wmf=WM_F*i, bmf=BM_F*i)
            c.dense2.inherit_and_evolve_WB(w2, b2, wmf=WM_F*i, bmf=BM_F*i)
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
    top = Car(0,0)
    original = Car(0,0)

    # Cars
    cars_crashed = 0
    cars = pg.sprite.Group()
    genCars(cars)

    # Evolution
    delta_time = 2000  # Time between checkpoints before force evolve
    generation = 1
    hs_w1 = np.arange(4, LAYER_NEURONS)
    hs_b1 = np.arange(1, LAYER_NEURONS)
    hs_w2 = np.arange(LAYER_NEURONS, 4)
    hs_b2 = np.arange(1, LAYER_NEURONS)

    while 1:
        # Input Events
        for event in pg.event.get():
            # Input Events
            # for i in cars:
                # if event.type == KEYDOWN:
                #     if event.key == K_w:
                #         i.accelerate = True
                #     if event.key == K_a:
                #         i.turn = -1
                #     if event.key == K_d:
                #         i.turn = 1
                #     if event.key == K_s:
                #         i.brake = True
                # if event.type == KEYUP:
                #     if event.key == K_w:
                #         i.accelerate = False
                #     if event.key == K_a:
                #         i.turn = 0
                #     if event.key == K_d:
                #         i.turn = 0
                #     if event.key == K_s:
                #         i.brake = False
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
                    hs_b1 = np.zeros_like(hs_b1)
                    hs_w2 = np.zeros_like(hs_w2)
                    hs_b2 = np.zeros_like(hs_b2)
                    genCars(cars)
                    original = cars.sprites()[0]
                if event.key == K_e:
                    # Force Evolve
                    cars.empty()
                    highest_score = 0
                    cars_crashed = 0
                    generation += 1
                    genCars(cars, w1=hs_w1, b1=hs_b1,
                            w2=hs_w2, b2=hs_b2, evolve=True)
                    original = cars.sprites()[0]
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
                if i.score <= highest_score-40:
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
                    # Score on checkpoint and lock
                    if not i.score_colliding and not i.anti_colliding:
                        check_time = pg.time.get_ticks()
                        i.score += 10
                        i.score_colliding = True
                if pg.sprite.collide_mask(i, anti_checkpoints):
                    # Checks if car is going wrong way
                    if not i.score_colliding:
                        i.crashed = True
                        i.accelerate = False
                        i.turn = False
                        i.brake = False
                        cars_crashed += 1
                    else:
                        i.anti_colliding = True
                if not pg.sprite.collide_mask(i, anti_checkpoints) and not pg.sprite.collide_mask(i, checkpoints) and i.anti_colliding:
                    # Checkpoint Unlock
                    i.anti_colliding = False
                    i.score_colliding = False
                # Gets highest score and saves car's data
                if i.score > highest_score:
                    try:
                        output = i.activation2.output
                    except:
                        pass
                    top = i
                    highest_score = i.score
                    hs_w1 = i.dense1.weights
                    hs_b1 = i.dense1.biases
                    hs_w2 = i.dense2.weights
                    hs_b2 = i.dense2.biases

                # Check for Sensors
                for s in range(i.sensors_amount):
                    if pg.sprite.collide_mask(i.sfront.sprites()[s], track):
                        if (s-1 < 0):
                            i.sfront_distance = (
                                i.sensors_len/i.sensors_amount)*(s+1)
                            i.sfront.sprites()[s].detection = True
                        elif not i.sfront.sprites()[s-1].detection:
                            i.sfront_distance = (
                                i.sensors_len/i.sensors_amount)*(s+1)
                            i.sfront.sprites()[s].detection = True
                    elif (s-1 >= 0):
                        if i.sfront.sprites()[s-1].detection:
                            i.sfront.sprites()[s].detection = True
                        if not i.sfront.sprites()[s-1].detection:
                            i.sfront.sprites()[s].detection = False
                    elif (s-1 < 0):
                        i.sfront.sprites()[s].detection = False
                    if (s == i.sensors_amount - 1) and not i.sfront.sprites()[s].detection:
                        i.sfront_distance = i.sensors_len + i.sensors_len/i.sensors_amount

                    if pg.sprite.collide_mask(i.sright.sprites()[s], track):
                        if (s-1 < 0):
                            i.sright_distance = (
                                i.sensors_len/i.sensors_amount)*(s+1)
                            i.sright.sprites()[s].detection = True
                        elif not i.sright.sprites()[s-1].detection:
                            i.sright_distance = (
                                i.sensors_len/i.sensors_amount)*(s+1)
                            i.sright.sprites()[s].detection = True
                    elif (s-1 >= 0):
                        if i.sright.sprites()[s-1].detection:
                            i.sright.sprites()[s].detection = True
                        if not i.sright.sprites()[s-1].detection:
                            i.sright.sprites()[s].detection = False
                    elif (s-1 < 0):
                        i.sright.sprites()[s].detection = False
                    if (s == i.sensors_amount - 1) and not i.sright.sprites()[s].detection:
                        i.sright_distance = i.sensors_len + i.sensors_len/i.sensors_amount

                    if pg.sprite.collide_mask(i.sleft.sprites()[s], track):
                        if (s-1 < 0):
                            i.sleft_distance = (
                                i.sensors_len/i.sensors_amount)*(s+1)
                            i.sleft.sprites()[s].detection = True
                        elif not i.sleft.sprites()[s-1].detection:
                            i.sleft_distance = (
                                i.sensors_len/i.sensors_amount)*(s+1)
                            i.sleft.sprites()[s].detection = True
                    elif (s-1 >= 0):
                        if i.sleft.sprites()[s-1].detection:
                            i.sleft.sprites()[s].detection = True
                        if not i.sleft.sprites()[s-1].detection:
                            i.sleft.sprites()[s].detection = False
                    elif (s-1 < 0):
                        i.sleft.sprites()[s].detection = False
                    if (s == i.sensors_amount - 1) and not i.sleft.sprites()[s].detection:
                        i.sleft_distance = i.sensors_len + i.sensors_len/i.sensors_amount
                
                if i.crashed:
                    cars.remove(i)

        if cars_crashed == GEN_SIZE:
            # Evolve after all cars crash
            cars.empty()
            highest_score = 0
            cars_crashed = 0
            generation += 1
            genCars(cars, w1=hs_w1, b1=hs_b1, w2=hs_w2, b2=hs_b2, evolve=True)
            original = cars.sprites()[0]

        if check_time <= pg.time.get_ticks()-delta_time:
            # Evolve after too much time passed between checkpoints
            check_time = pg.time.get_ticks()
            cars.empty()
            highest_score = 0
            cars_crashed = 0
            generation += 1
            genCars(cars, w1=hs_w1, b1=hs_b1, w2=hs_w2, b2=hs_b2, evolve=True)
            original = cars.sprites()[0]

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
                    for s in range(i.sensors_amount):
                        screen.blit(i.sfront.sprites()[
                                    s].image, i.sfront.sprites()[s].rect.topleft)
                        screen.blit(i.sleft.sprites()[
                                    s].image, i.sleft.sprites()[s].rect.topleft)
                        screen.blit(i.sright.sprites()[
                                    s].image, i.sright.sprites()[s].rect.topleft)

        screen.blit(pg.font.Font.render(font, "Highest Score: " +
                                        str(highest_score), True, (255, 255, 255)), (5, 10))
        screen.blit(pg.font.Font.render(font, "Crashed: " +
                                        str(cars_crashed) + "/" + str(GEN_SIZE), True, (255, 255, 255)), (250, 10))
        screen.blit(pg.font.Font.render(font, "Generation: " +
                                        str(generation), True, (255, 255, 255)), (420, 10))
        screen.blit(pg.font.Font.render(font, "Time between checkpoints: " +
                                        str((pg.time.get_ticks() - check_time)/1000), True, (255, 255, 255)), (600, 10))

        if top in cars:
            screen.blit(pg.font.Font.render(font, "1", True, (255, 255, 255)), top.rect.topleft)
        if original in cars:
            screen.blit(pg.font.Font.render(font, "og", True, (255, 255, 255)), original.rect.topright)

        pg.display.flip()

        clock.tick(144)


if __name__ == "__main__":
    main()
