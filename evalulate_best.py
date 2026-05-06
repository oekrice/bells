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
from learn import run_bell

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

Net = ForceNet(2, 6)
Net.generate_random_seed()

#nets = Networks()  #This is the old networks one

strike_limit = 1.0

max_time = 60.0
mode = 'up'
load_best = True
extend_net = False


def evaluate_theta(theta, angles):
    global mode

    total_fitness = 0.0
    simulation_seconds = 60.0

    for initial_angle in angles:
        #Perhaps start off near the top and gradually work down? Worth a shot.

        phy = init_physics()
        phy.do_volume = False

        #bell = initialise_bell(phy, initial_angle, 0.0)
        sim = run_bell()  # all the physics in here

        wheel_force = 600  # Max. force on the rope (in Newtons)
        count = 0
        fitness = 0.0

        ring_up = False
        ring_down = False
        ring_steady = False

        sim.bell.current_mode = mode

        Net.update_network(theta)

        # Check for inputs that affect the timestep
        force = 0.0  # This value between 0 and 1 and then update based on the physics.

        #amin = 0.0*np.pi; amax = 0.1*np.pi

        sim.bell.bell_angle = initial_angle# uniform(amin, amax)

        sim.bell.clapper_angle = np.sign(sim.bell.bell_angle)*sim.bell.clapper_limit + sim.bell.bell_angle

        sim.bell.stay_break_limit = 0.4

        sim.bell.velocity = 0.0

        if np.abs(sim.bell.bell_angle) < 0.5:
            sim.bell.max_length = 0.0  # max backstroke length
        else:
            sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.phy.time < simulation_seconds:

            inputs = sim.bell.get_scaled_state()

            # Inputs are the things we can know -- in my case it is the angle and speed of the bell (for now)
            # Do try to remember to get inputs in the range (0,1). Can do easily enough.
            # This is just a list.
            # Apply action to the simulated cart-pole
            if sim.bell.current_mode == 'up':
                ring_up = True
                action = Net.force(inputs)
                force = min(1.0, action[0])

            if sim.bell.current_mode == 'down':
                ring_down = True
                action = Net.force(inputs)
                force = min(1.0, action[0])

            if sim.bell.current_mode == 'steady':
                ring_steady = True
                action = Net.force(inputs)
                force = min(1.0, action[0])

            sim.bell.pull = force
            sim.step(force)

            fitness = fitness + sim.bell.fitness_increment(sim.phy)*(simulation_seconds)/(simulation_seconds)

            phy.count = phy.count + 1

        # Check for force on wheel - this takes effect at the next timestep
        # Check for actions or stay smash. All needs to be in the same event.get for some reason.

        if sim.bell.stay_hit > 0:
            sim.bell.stay_angle = 1e6
            fitness = 1e9#fitness*10.0  #Stay break penalty (quite extreme)
            #print('Stay broken')
            break

        count += 1

        #print(fitness, phy.time, bell.bell_angle, bell.velocity)
        total_fitness += fitness

    return sim.bell.bell_angles, sim.bell.velocities

if load_best:
    Net.load_latest_state(mode)
    print('Loaded best state')
else:
    Net.generate_random_seed()
    print('Generated random state')

go = True
while go:

    data_length = 0
    fname = f'./nets/{mode}.txt'

    if os.path.exists(fname):
        best_score = 1e6; best_id = 0
        specific_parameters = []
        with open(fname, "r") as f:
            data = f.readlines()
        for i in range(len(data)):
            score = float(data[i].split(" ")[1])
            if score < best_score:
                best_score = score
                best_id = i

    best_id = len(data) - 1
    success = False
    while not success:
        try:
            Net.load_specific_state(mode,best_id)
            success = True
        except:
            time.sleep(1.0)

    print('Loading state', best_id)
    score = data[best_id].split(" ")[1]
    log_num  = best_id
    print('Score:', score)
    #Net.load_best_state(mode,log_num)

    if extend_net:
        print(f'Extending net to {n_nodes_target} nodes')
        Net.extend_net(n_nodes_target=n_nodes_target)
        n_nodes = n_nodes_target

    if False:
        fitness = evaluate_theta(Net.parameter_set, [0.0])

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

        bell_angles, bell_velocities = evaluate_theta(Net.parameter_set, [0.0])

        plt.plot(bell_angles, bell_velocities, c = 'red')
        #Attempt a colourmap?
        angles = np.linspace(-np.pi-0.15, np.pi+0.15,250)
        velocities = np.linspace(-10,10,250)
        # cmap = np.zeros((len(angles), len(velocities)))
        # for i, angle in enumerate(angles):
        #     for j, velocity in enumerate(velocities):
        #         cmap[i,j] = Net.force([angle,velocity])[0]
        # plt.pcolormesh(angles, velocities, cmap.T)
        plt.xlabel('Bell angle')
        plt.ylabel('Bell velocity')
        #plt.colorbar()
        plt.title(f'{log_num}, {score}')
        plt.tight_layout()
        plt.savefig('./plots/%d_cmap.png' % log_num)
        plt.close()
    go = False




#run_cma_mp(n_cores=1)

#evaluate_theta(Net.parameter_set)
#evaluate_theta(Net.parameter_set)




