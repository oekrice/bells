# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:03:56 2024

@author: eleph
"""

import asyncio

# import nest_asyncio

import pygame, sys
from pygame.locals import *
import numpy as np
import neat
import pickle
import os
import random

from bell_physics import init_bell, init_physics
from display import display_tools

if False:
    nest_asyncio.apply()

pygame.init()

phy = init_physics()
bell = init_bell(phy, 0.0)
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
        self.up = neat.nn.FeedForwardNetwork.create(up, config)
        with open("networks/ring_down", "rb") as f:
            down = pickle.load(f)
        self.down = neat.nn.FeedForwardNetwork.create(down, config)


nets = Networks()

if True:
    #Find current best ringing up
    os.system('scp best_so_far ./networks/ring_down')

async def main():

    fpsClock = pygame.time.Clock()

    wheel_force = 600  # force on the rope (in Newtons)
    count = 0
    ring_up = False
    ring_down = False

    while True:  # the main game loop

        # Check for inputs that affect the timestep
        press_keys = pygame.key.get_pressed()
        press_mouse = pygame.mouse.get_pressed()

        force = 0.0   #This value between 0 and 1 and then update based on the physics

        if press_keys[pygame.K_SPACE] or press_mouse[0]:
            force = 1.0

        if ring_up:
            inputs = bell.get_scaled_state()
            action = nets.up.activate(inputs)
            force = min(1.0, force+action[0])

        if ring_down:
            inputs = bell.get_scaled_state()
            action = nets.down.activate(inputs)
            force = min(1.0, force+action[0])

        if bell.effect_force < 0.0:  # Can pull the entire handstroke
            bell.wheel_force = force*bell.effect_force * wheel_force
        else:  # Can only pull some of the backstroke
            if bell.rlength > bell.max_length - bell.backstroke_pull:
                bell.wheel_force = force*bell.effect_force * wheel_force
            else:
                bell.wheel_force = force*0.0

        dp.surface.fill(dp.WHITE)

        bell.timestep(phy)

        if phy.count > 2:
            fitness = bell.fitness_fn()

        phy.count = phy.count + 1

        dp.draw_rope(phy, bell)

        dp.draw_bell(phy, bell)

        dp.display_stroke(phy, bell)  # Displays the text 'handstroke' or 'backstroke'

        dp.display_state(phy, ring_up, ring_down)

        dp.display_force(phy, bell.wheel_force)
        # Check for sound
        if bell.ding == True:
            # if abs(bell.bell_angle) > bell.sound_angle and abs(bell.prev_angle) <= bell.sound_angle:
            bell.sound.play()
            # continue
        # Check for force on wheel - this takes effect at the next timestep

        mouse = pygame.mouse.get_pos()  # use to activate things

        # Check for actions or stay smash. All needs to be in the same event.get for some reason.
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    ring_up = not (ring_up)
                    ring_down = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    ring_down = not (ring_down)
                    ring_up = False

            if event.type == 1025:
                if mouse[0] > 40 and mouse[0] < 110 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_up = not (ring_up)
                    ring_down = False

            if event.type == 1025:

                if mouse[0] > 270 and mouse[0] < 340 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_down = not (ring_down)
                    ring_up = False

            if event.type == QUIT:
                pygame.quit()
                return

        if bell.stay_hit > 0:
            bell.stay_angle = 1e6

        pygame.display.update()
        fpsClock.tick(phy.FPS)

        await asyncio.sleep(0)


asyncio.run(main())
