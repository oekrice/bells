# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:03:56 2024

@author: eleph
"""

import asyncio

import nest_asyncio

import pygame, sys
from pygame.locals import *
import numpy as np
import neat
import pickle
import os
import random
from random import uniform, gauss
import sys

from bell_physics import init_bell, init_physics
from display import display_tools

if True:
    nest_asyncio.apply()

if len(sys.argv) > 1:
    load_num = int(sys.argv[1])
else:
    load_num = -1

pygame.init()

phy = init_physics()
bell = init_bell(phy, 0.0)

runs = 15; runs_per_net = 30
amin = np.pi * 0.9; amax = np.pi
rmin = (runs//2)*(amax - amin)/(runs_per_net//2) + amin
rmax = (runs//2+1)*(amax - amin)/(runs_per_net//2) + amin

if runs%2 == 0:
    rmin = -rmin; rmax = -rmax

bell.bell_angle = 0.0#uniform(rmin, rmax)
bell.clapper_angle = np.sign(bell.bell_angle)*bell.clapper_limit + bell.bell_angle

if np.abs(bell.bell_angle) < 0.5:
    bell.max_length = 0.0  # max backstroke length
else:
    bell.max_length = bell.radius*(1.0 + 3*np.pi/2 - bell.garter_hole)


bell.target_period = 5.0
bell.stay_break_limit = 1.0

bell.m_1 = 500
bell.m_2 = 0.05*bell.m_1

print('Bell mass', bell.m_1)

dp = display_tools(phy, bell)

bell.sound = pygame.mixer.Sound("bellsound_deep.wav")

# Set up colours
dp.define_colours()
# Import images and transform scales
dp.import_images(phy, bell)
# set up the window
pygame.display.set_caption("Animation")

class Networks:
    def __init__(self):
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "networks/config")
        config = neat.Config(
            neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path
        )
        with open("networks/ring_up", "rb") as f:
            up = pickle.load(f)
            print(up)
        self.up = neat.nn.FeedForwardNetwork.create(up, config)
        with open("networks/ring_down", "rb") as f:
            down = pickle.load(f)
        self.down = neat.nn.FeedForwardNetwork.create(down, config)
        with open("networks/ring_steady", "rb") as f:
            steady = pickle.load(f)
        self.steady = neat.nn.FeedForwardNetwork.create(steady, config)

if True:
    # Find current best ringing up
    if load_num < 0:
        os.system("scp current_best ./networks/ring_up")
    else:
        os.system("scp ./current_network/%d ./networks/ring_up" % load_num)
if False:
    # Find current best ringing up
    if load_num < 0:
        os.system("scp current_best ./networks/ring_down")
    else:
        os.system("scp ./current_network/%d ./networks/ring_down" % load_num)
if False:
    # Find current best ringing up
    if load_num < 0:
        os.system("scp current_best ./networks/ring_steady")
    else:
        os.system("scp ./current_network/%d ./networks/ring_steady" % load_num)


nets = Networks()

refresh_rate = 2

strike_limit = 1.0
async def main():

    fpsClock = pygame.time.Clock()

    wheel_force = 600  # force on the rope (in Newtons)
    count = 0
    ring_up = True
    ring_down = False
    ring_steady = False
    dp.surface.fill(dp.WHITE)

    while True:  # the main game loop

        # Check for inputs that affect the timestep
        press_keys = pygame.key.get_pressed()
        press_mouse = pygame.mouse.get_pressed()

        force = 0.0  # This value between 0 and 1 and then update based on the physics

        if press_keys[pygame.K_SPACE] or press_mouse[0]:
            force = 1.0

        if ring_up:
            inputs = bell.get_scaled_state()[:2]
            action = nets.up.activate(inputs)
            force = min(1.0, force + action[0])

        if ring_down:
            inputs = bell.get_scaled_state()[:2]
            action = nets.down.activate(inputs)
            force = min(1.0, force + action[0])

        if ring_steady:
            inputs = bell.get_scaled_state()
            action = nets.steady.activate(inputs)
            force = min(1.0, force + action[0])

        if bell.stay_hit > 0:
            force = 0.0


        if bell.effect_force < 0.0:  # Can pull the entire handstroke
            bell.wheel_force = force * bell.effect_force * wheel_force
        else:  # Can only pull some of the backstroke
            if bell.rlength > bell.max_length - bell.backstroke_pull:
                bell.wheel_force = force * bell.effect_force * wheel_force
            else:
                bell.wheel_force = force * 0.0

        bell.pull = force

        phy.count = phy.count + 1

        if count % refresh_rate == 0:

            dp.surface.fill(dp.WHITE)

            dp.draw_rope(phy, bell)

            if count % refresh_rate * 3 == 0:

                dp.display_stroke(phy, bell)  # Displays the text 'handstroke' or 'backstroke'

                dp.display_state(phy, bell, ring_up, ring_down, ring_steady)

                dp.display_force(phy, bell, bell.wheel_force)

            dp.draw_bell(phy, bell)

        # Check for sound
        if bell.ding == True:
            # if abs(bell.bell_angle) > bell.sound_angle and abs(bell.prev_angle) <= bell.sound_angle:
            bell.sound.play()
            if bell.bell_angle > 0:
                print('Back', bell.backstroke_target)
            else:
                print('Hand', bell.handstroke_target)
            # continue
        # Check for force on wheel - this takes effect at the next timestep

        mouse = pygame.mouse.get_pos()  # use to activate things

        #print(bell.handstroke_targets, bell.backstroke_targets)
        # Check for actions or stay smash. All needs to be in the same event.get for some reason.
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    ring_up = not (ring_up)
                    ring_down = False
                    ring_steady = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    ring_down = not (ring_down)
                    ring_up = False
                    ring_steady = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    ring_steady = not (ring_steady)
                    ring_up = False
                    ring_down = False
                    bell.update_rhythm = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    bell.target_period = bell.target_period - 0.1
                    bell.update_rhythm = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    bell.target_period = bell.target_period + 0.1
                    bell.update_rhythm = True

            if event.type == 1025:
                if mouse[0] > 40 and mouse[0] < 110 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_up = not (ring_up)
                    ring_down = False
                    ring_steady = False

            if event.type == 1025:

                if mouse[0] > 270 and mouse[0] < 340 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_down = not (ring_down)
                    ring_up = False
                    ring_steady = False

            if event.type == 1025:
                if bell.stay_hit > 0:
                    if mouse[1] > 0.8 * phy.pixels_y:
                        bell.bell_angle = 0.0
                        bell.clapper_angle = 0.0
                        bell.velocity = 0.0
                        bell.clapper_velocity = 0.0
                        bell.stay_hit = 0
                        bell.prev_angle = 0.0
                        bell.max_length = 0.0  # max backstroke length
                        bell.stay_angle = 0.15


            if event.type == QUIT:
                pygame.quit()
                return

        bell.timestep(phy)

        '''
        if len(bell.backstroke_accuracy) > 0:
            print(bell.backstroke_accuracy[-1])
        if len(bell.handstroke_accuracy) > 0:
            print(bell.handstroke_accuracy[-1])
        '''

        if bell.stay_hit > 0:
            bell.stay_angle = 1e6

        if count % refresh_rate == 0:
            pygame.display.update()
        if count % 60 == 0:
            #fitness = bell.fitness_fn(phy, print_accuracy = True)
            print(bell.fitness_increment(phy)*60*60)
            print('Time', phy.time, 'Angle', bell.bell_angle)
            #rpint('Fitness', fitness)
            #print(bell.handstroke_accuracy)
            #print(bell.backstroke_accuracy)

        #'Learn as it goes'
        if count % (60*60) == -1:
            # Find current best ringing up
            if load_num >= 0:
                os.system("scp ./current_network/%d ./networks/ring_up" % load_num)
            else:
                for i in range(10000):
                    if not os.path.isfile('./current_network/%d' % (i+1)):
                        break

                os.system("scp ./current_network/%d ./networks/ring_up" % i)

            #nets = Networks()

        count += 1

        fpsClock.tick(phy.FPS)

        await asyncio.sleep(0)


asyncio.run(main())
