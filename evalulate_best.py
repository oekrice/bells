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

n_nodes = 20
n_inputs = 6


def evaluate_theta(theta):
    global mode

    end_height = np.pi+0.125
    n_angles = 51

    angles = np.linspace(-end_height, end_height, n_angles)
    #angles = [3.0]
    total_fitness = 0.0
    all_fitnesses = []

    for init_angle in angles:

        print('Init angle', init_angle)
        wheel_force = 600  # Max. force on the rope (in Newtons)
        count = 0
        fitness = 0.0

        ring_up = False
        ring_down = False
        ring_steady = False

        Net_local = ForceNet(n_nodes, n_inputs)

        Net_local.update_network(theta)

        sim = run_bell()

        sim.bell.current_mode = mode

        sim.bell.bell_angle = init_angle

        sim.bell.clapper_angle = np.sign(sim.bell.bell_angle)*sim.bell.clapper_limit + sim.bell.bell_angle

        sim.bell.stay_break_limit = 0.25

        sim.bell.velocity = 0.0

        if np.abs(sim.bell.bell_angle) < 0.5:
            sim.bell.max_length = 0.0  # max backstroke length
        else:
            sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)

        # Run the given simulation for up to num_steps time steps.

        while sim.phy.time < max_time:
            force = 0.0  # This value between 0 and 1 and then update based on the physics.

            inputs = sim.bell.get_scaled_state()[:n_inputs]

            if sim.bell.current_mode == 'up':
                ring_up = True
                action = Net_local.force(inputs)
                force = min(1.0, action[0])

            if sim.bell.current_mode == 'down':
                ring_down = True
                action = Net_local.force(inputs)
                force = min(1.0, action[0])

            if sim.bell.current_mode == 'steady':
                ring_steady = True
                action = Net_local.force(inputs)
                force = min(1.0, action[0])

            sim.bell.pull = force
            sim.step(force)

            #fitness = fitness + sim.bell.fitness_increment(sim.phy)

            sim.phy.count = sim.phy.count + 1

            if sim.bell.stay_touch > 0:
                break

        fitness = sim.bell.fitness_fn(sim.phy, verbose=True)

        # if sim.bell.stay_hit > 0:
        #     sim.bell.stay_angle = 1e6
        #     fitness = 1.0
        #
        #total_fitness += fitness
        all_fitnesses.append(fitness)


        #print('Fitness for angle:', init_angle, fitness)
        c = 'black'
        if sim.bell.bell_angles[-1] > np.pi and sim.bell.stay_touch_velocity < 0.25:
            c = 'green'
        elif sim.bell.bell_angles[-1] > np.pi:
            c = 'red'
        elif sim.bell.bell_angles[-1] < -np.pi:
            c = 'red'

        cut = len(sim.bell.bell_angles)
        plt.plot(sim.bell.times[:cut], sim.bell.bell_angles,linewidth=0.5,c=c)
        #plt.plot(bell.forces)
    plt.show()
    alpha = 4
    all_fitnesses = np.array(all_fitnesses)
    total_fitness = (np.sum(all_fitnesses**alpha)/len(angles))**(1.0/alpha)

    return total_fitness

max_time = 10.0
mode = 'up'
load_best = True
extend_net = True

Net = ForceNet(n_nodes, n_inputs)

if load_best:
    Net.load_best_state(mode, override_nnodes=True, latest=True)
    print('Loaded best state')
else:
    Net.generate_random_seed()
    print('Generated random state')

#nets = Networks()  #This is the old networks one


if False:
    while True:
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
                    if float(line.split(' ')[1]) < 100.0:
                        best_scores.append(best_score)
                        scores.append(float(line.split(' ')[1]))
        plt.plot(scores)
        plt.xscale('log')
        plt.yscale('log')
        plt.savefig('./plots/best_score.png')
        #plt.show()
        plt.close()
        time.sleep(5.0)

if True:
    fitness = evaluate_theta(Net.parameter_set)
    print(fitness)

elif False:
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




