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

import matplotlib.pyplot as plt

from bell_physics import init_bell, init_physics
from display import display_tools
from nets import ForceNet

import cma
import os
import multiprocessing as mp
import time

if True:
    nest_asyncio.apply()

if len(sys.argv) > 1:
    load_num = int(sys.argv[1])
else:
    load_num = -1


audio_enabled = False

phy = init_physics()
phy.do_volume = False

def initialise_bell(phy, angle=0.0, velocity = 0.0):

    bell = init_bell(phy, 0.0)

    bell.bell_angle = angle#0.0#uniform(rmin, rmax)
    bell.velocity = velocity
    bell.clapper_angle = np.sign(bell.bell_angle)*bell.clapper_limit + bell.bell_angle

    if np.abs(bell.bell_angle) < 0.5:
        bell.max_length = 0.0  # max backstroke length
    else:
        bell.max_length = bell.radius*(1.0 + 3*np.pi/2 - bell.garter_hole)

    bell.target_period = 5.0
    bell.stay_break_limit = 1.0

    bell.m_1 = 500   #Bell mass
    bell.m_2 = 0.05*bell.m_1   #Clapper mass

    return bell

Net = ForceNet(4, 2)
Net.generate_random_seed()

#nets = Networks()  #This is the old networks one

strike_limit = 1.0

max_time = 120.0
mode = 'up'
load_best = True
extend_net = False

def evaluate_theta(theta):
    global mode

    angles = np.linspace(-np.pi-0.1, np.pi+0.1, 11)
    total_fitness = 0.0

    for init_angle in angles:

        phy = init_physics()
        phy.do_volume = False

        bell = initialise_bell(phy, init_angle, 0.0)

        wheel_force = 600  # Max. force on the rope (in Newtons)
        count = 0
        fitness = 0.0

        ring_up = False
        ring_down = False
        ring_steady = False

        bell.current_mode = mode

        Net.update_network(theta)

        while phy.time < max_time:  # the main game loop
            t0 = time.time()
            # Check for inputs that affect the timestep
            force = 0.0  # This value between 0 and 1 and then update based on the physics

            inputs = bell.get_scaled_state()

            if bell.current_mode == 'up':
                ring_up = True
                action = Net.force(inputs[:2])
                force = min(1.0, force + action[0])

            if bell.current_mode == 'down':
                ring_down = True
                action = Net.force(inputs[:2])
                force = min(1.0, force + action[0])

            if bell.current_mode == 'steady':
                ring_steady = True
                action = Net.force(inputs)
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

            # Check for force on wheel - this takes effect at the next timestep

            #print(bell.handstroke_targets, bell.backstroke_targets)
            # Check for actions or stay smash. All needs to be in the same event.get for some reason.

            bell.timestep(phy)
            fitness += bell.fitness_increment(phy)

            if bell.stay_hit > 0:
                bell.stay_angle = 1e6


            # if count % 60 == 0:
            #     #fitness = bell.fitness_fn(phy, print_accuracy = True)
            #     print(bell.fitness_increment(phy)*60*60)
            #     print('Time', phy.time, 'Angle', bell.bell_angle)

            count += 1

        plt.plot(bell.bell_angles)
        #plt.plot(bell.forces)
    plt.show()

    print('Fitness', fitness)
    return fitness

if load_best:
    Net.load_best_state(mode, override_nnodes=True)
    print('Loaded best state')
else:
    Net.generate_random_seed()
    print('Generated random state')

if extend_net:
    print(f'Extending net to {n_nodes_target} nodes')
    Net.extend_net(n_nodes_target=n_nodes_target)
    n_nodes = n_nodes_target

if True:
    fitness = evaluate_theta(Net.parameter_set)

elif False:
    #Plot best scores.
    fname = f'./nets/{mode}.txt'
    scores = []; best_scores = []
    #Determine the correct number of parameters for this best state
    if os.path.exists(fname):
        best_score = 1e6; best_id = 0
        best_parameters = []
        with open(fname, "r") as f:
            data = f.readlines()
            cut = len(data)
            for li, line in enumerate(data[-cut:]):
                if float(line.split(' ')[1]) < best_score:
                    best_score = float(line.split(' ')[1])
                    best_id = len(data) - cut + li
                if float(line.split(' ')[1]) < 1000.0:
                    best_scores.append(best_score)
                    scores.append(float(line.split(' ')[1]))
    plt.plot(scores)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()

elif True:
    #Attempt a colourmap?
    angles = np.linspace(-np.pi-0.1, np.pi+0.1,250)
    velocities = np.linspace(-10,10,250)
    cmap = np.zeros((len(angles), len(velocities)))
    for i, angle in enumerate(angles):
        for j, velocity in enumerate(velocities):
            cmap[i,j] = Net.force([angle,velocity])[0]
    plt.pcolormesh(angles, velocities, cmap.T)
    plt.xlabel('Bell angle')
    plt.ylabel('Bell velocity')
    plt.colorbar()
    plt.show()





#run_cma_mp(n_cores=1)

#evaluate_theta(Net.parameter_set)
#evaluate_theta(Net.parameter_set)




