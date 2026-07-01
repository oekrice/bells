# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:03:56 2024

@author: eleph
"""

import asyncio

#import nest_asyncio

import pygame, sys
from pygame.locals import QUIT, Color
import numpy as np
import pickle
import os
import random
from random import uniform, gauss
import sys

from bell_physics import init_bell, init_physics
from display import display_tools
from nets import ForceNet
from learn import run_bell

if False:
    nest_asyncio.apply()



async def main():

    print('Thing is running!')

    pygame.init()

    try:
        pygame.mixer.init()
        audio_enabled = True
    except pygame.error:
        print("Audio disabled")
        audio_enabled = False

    print('Thing is running!')
    sim = run_bell()

    dp = display_tools(sim.phy, sim.bell)

    if audio_enabled:
        sim.bell.sound = pygame.mixer.Sound("bellsound_deep.ogg")
    else:
        sim.bell.sound = None
        sim.phy.do_volume = False

    # Set up colours
    dp.define_colours()
    # Import images and transform scales
    dp.import_images(sim.phy, sim.bell)
    # set up the window
    pygame.display.set_caption("Animation")

    n_inputs = 13
    refresh_rate = 2
    n_nodes = 50

    Net = ForceNet(n_nodes, n_inputs)
    Net.generate_random_seed()

    fpsClock = pygame.time.Clock()

    wheel_force = 600  # Max. force on the rope (in Newtons)
    count = 0
    fitness = 0.0

    ring_up = False
    ring_down = False
    ring_steady = False
    ring_up_back = False

    dp.surface.fill(dp.WHITE)

    init_angle = 0.0

    sim.bell.bell_angle = init_angle

    sim.bell.clapper_angle = np.sign(sim.bell.bell_angle)*sim.bell.clapper_limit + sim.bell.bell_angle

    sim.bell.stay_break_limit = 100.0

    sim.bell.velocity = 0.0

    sim.bell.m_1 = 300
    sim.bell.m_2 = 0.05*sim.bell.m_1

    sim.bell.timing_seeds = 0.0*sim.bell.timing_seeds

    if np.abs(sim.bell.bell_angle) < 0.5:
        sim.bell.max_length = 0.0  # max backstroke length
    else:
        sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)

    Net = ForceNet(50, n_inputs)

    while True:  # the main game loop

        # Check for inputs that affect the timestep
        press_keys = pygame.key.get_pressed()
        press_mouse = pygame.mouse.get_pressed()

        force = 0.0  # This value between 0 and 1 and then update based on the physics

        if press_keys[pygame.K_SPACE] or press_mouse[0]:
            force = 1.0

        inputs = sim.bell.get_scaled_state()[:n_inputs]

        if sim.bell.current_mode == 'up':
            ring_up = True
            action = Net.force(inputs)
            force = min(1.0, action[0] + force)
        if sim.bell.current_mode == 'down':
            ring_down = True
            action = Net.force(inputs)
            force = min(1.0, action[0] + force)

        if sim.bell.current_mode == 'steady':
            ring_steady = True
            action = Net.force(inputs)
            force = min(1.0, action[0] + force)

        if sim.bell.current_mode == 'up_back':
            ring_up_back = True
            action = Net.force(inputs)
            force = min(1.0, action[0] + force)

        sim.bell.pull = force

        if count % refresh_rate == 0:

            dp.surface.fill(dp.WHITE)

            dp.draw_rope(sim.phy, sim.bell)

            if count % refresh_rate * 3 == 0:

                dp.display_stroke(sim.phy, sim.bell)  # Displays the text 'handstroke' or 'backstroke'

                dp.display_state(sim.phy, sim.bell, ring_up, ring_down, ring_steady, ring_up_back)

                dp.display_force(sim.phy, sim.bell, sim.bell.wheel_force)

                dp.display_mass(sim.phy, sim.bell)

            dp.draw_bell(sim.phy, sim.bell)

        # Check for sound
        if sim.bell.ding == True:
            # if abs(bell.bell_angle) > bell.sound_angle and abs(bell.prev_angle) <= bell.sound_angle:
            if audio_enabled:
                sim.bell.sound.play()

            # continue
        # Check for force on wheel - this takes effect at the next timestep

        mouse = pygame.mouse.get_pos()  # use to activate things

        if True and count%(60*60) == 1 and sim.bell.current_mode == 'steady':  #Learn as it goes
            Net.load_best_state('steady' , override_nnodes=True, latest=True)

        #Introduce a check for when the bell is sucessfully up, and stop when it gets there?
        #print(bell.handstroke_targets, bell.backstroke_targets)
        # Check for actions or stay smash. All needs to be in the same event.get for some reason.
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_u:
                    ring_up = not (ring_up)
                    ring_down = False
                    ring_steady = False
                    ring_up_back = False
                    if sim.bell.current_mode == 'up':
                        sim.bell.current_mode = 'none'
                    else:
                        sim.bell.current_mode = 'up'
                        #Net.update_network(best_theta_up)
                        Net = ForceNet(50, n_inputs)
                        Net.load_best_state('up', override_nnodes=True, latest=False, bestever=False)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    ring_down = not (ring_down)
                    ring_up = False
                    ring_steady = False
                    ring_up_back = False

                    if sim.bell.current_mode == 'down':
                        sim.bell.current_mode = 'none'
                    else:
                        sim.bell.current_mode = 'down'
                        Net = ForceNet(50, n_inputs)
                        Net.load_best_state('down', override_nnodes=True, latest=False, bestever=False)

            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_s:
            #         ring_steady = not (ring_steady)
            #         ring_up = False
            #         ring_down = False
            #         ring_up_back = False
            #         sim.bell.update_rhythm = True
            #         if sim.bell.current_mode == 'steady':
            #             sim.bell.current_mode = 'none'
            #         else:
            #             sim.bell.current_mode = 'steady'
            #             sim.bell.strict_rhythm = False

            #             Net = ForceNet(50, n_inputs)
            #             Net.load_best_state('steady', override_nnodes=True, latest=False, bestever=False)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_y:
                    ring_up_back = not (ring_up_back)
                    ring_up = False
                    ring_down = False
                    ring_steady = False
                    sim.bell.update_rhythm = True
                    if sim.bell.current_mode == 'up_back':
                        sim.bell.current_mode = 'none'
                    else:
                        sim.bell.current_mode = 'up_back'
                        Net = ForceNet(50, n_inputs)
                        Net.load_best_state('up_back', override_nnodes=True, latest=False, bestever=False)


            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_z:
                    sim.bell.target_period = sim.bell.target_period - 0.1
                    sim.bell.update_rhythm = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    sim.bell.target_period = sim.bell.target_period + 0.1
                    sim.bell.update_rhythm = True

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o:
                    sim.bell.m_1 = sim.bell.m_1 - 10
                    sim.bell.m_2 = 0.05*sim.bell.m_1
                    print(f'Bell mass = {sim.bell.m_1}')

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    sim.bell.m_1 = sim.bell.m_1 + 10
                    sim.bell.m_2 = 0.05*sim.bell.m_1
                    print(f'Bell mass = {sim.bell.m_1}')

            if event.type == 1025:
                if mouse[0] > 40 and mouse[0] < 75 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_up = not (ring_up)
                    ring_up_back = False
                    ring_down = False
                    ring_steady = False
                    if sim.bell.current_mode == 'up':
                        sim.bell.current_mode = 'none'
                    else:
                        sim.bell.current_mode = 'up'
                        #Net.update_network(best_theta_up)
                        Net = ForceNet(50, n_inputs)
                        Net.load_best_state('up', override_nnodes=True, latest=False, bestever=False)

            if event.type == 1025:
                if mouse[0] > 75 and mouse[0] < 110 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_up = False
                    ring_down = False
                    ring_steady = False
                    ring_up_back = not (ring_up_back)
                    if sim.bell.current_mode == 'up_back':
                        sim.bell.current_mode = 'none'
                    else:
                        sim.bell.current_mode = 'up_back'
                        #Net.update_network(best_theta_up)
                        Net = ForceNet(50, n_inputs)
                        Net.load_best_state('up_back', override_nnodes=True, latest=False, bestever=False)

            if event.type == 1025:
                if mouse[0] > 270 and mouse[0] < 340 and mouse[1] > 70 and mouse[1] < 90:
                    # left button
                    ring_down = not (ring_down)
                    ring_up = False
                    ring_steady = False
                    ring_up_back = False

                    if sim.bell.current_mode == 'down':
                        sim.bell.current_mode = 'none'
                    else:
                        sim.bell.current_mode = 'down'
                        #Net.update_network(best_theta_up)
                        Net = ForceNet(50, n_inputs)
                        Net.load_best_state('down', override_nnodes=True, latest=False, bestever=False)

            if event.type == 1025:
                if sim.bell.stay_hit > 0:
                    if mouse[1] > 0.8 * sim.phy.pixels_y:
                        sim.bell.bell_angle = 0.0
                        sim.bell.clapper_angle = 0.0
                        sim.bell.velocity = 0.0
                        sim.bell.clapper_velocity = 0.0
                        sim.bell.stay_hit = 0
                        sim.bell.prev_angle = 0.0
                        sim.bell.max_length = 0.0  # max backstroke length
                        sim.bell.stay_angle = 0.15

            if event.type == 1025:
                if mouse[0] > 0.65 * sim.phy.pixels_x and mouse[0] < 0.69 * sim.phy.pixels_x  and mouse[1] > 0.9 * sim.phy.pixels_y:
                    # Change bell mass downwards
                    if sim.bell.m_1 >  50:
                        sim.bell.m_1 = sim.bell.m_1 - 10
                        sim.bell.m_2 = 0.05*sim.bell.m_1

            if event.type == 1025:
                if mouse[0] > 0.74 * sim.phy.pixels_x and mouse[0] < 0.78 * sim.phy.pixels_x  and mouse[1] > 0.9 * sim.phy.pixels_y:
                    # Change bell mass upwards
                    if sim.bell.m_1 < 500:
                        sim.bell.m_1 = sim.bell.m_1 + 10
                        sim.bell.m_2 = 0.05*sim.bell.m_1



            if event.type == QUIT:
                pygame.quit()
                return

        #bell_energy = sim.bell.bell_angle**2 + sim.bell.velocity**2

        sim.step(force)

                #fitness += bell.fitness_increment(phy)
        '''
        if len(bell.backstroke_accuracy) > 0:
            print(bell.backstroke_accuracy[-1])
        if len(bell.handstroke_accuracy) > 0:
            print(bell.handstroke_accuracy[-1])
        '''

        if sim.bell.stay_hit > 0:
            sim.bell.stay_angle = 1e6

        if count % refresh_rate == 0:
            pygame.display.update()
        # if count % 60 == 0:
        #     #fitness = bell.fitness_fn(phy, print_accuracy = True)
        #     print(bell.fitness_increment(phy)*60*60)
        #     print('Time', phy.time, 'Angle', bell.bell_angle)
        #     #rpint('Fitness', fitness)
        #     #print(bell.handstroke_accuracy)
        #     #print(bell.backstroke_accuracy)


            #nets = Networks()

        count += 1

        fpsClock.tick(sim.phy.FPS)

        await asyncio.sleep(0)


asyncio.run(main())
