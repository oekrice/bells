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

from scipy.stats.qmc import LatinHypercube as hpc
import matplotlib
matplotlib.use('Agg')

if True:
    nest_asyncio.apply()

if len(sys.argv) > 1:
    load_num = int(sys.argv[1])
else:
    load_num = -1


audio_enabled = False

phy = init_physics()
phy.do_volume = False

n_nodes = 50
n_inputs = 13


def evaluate_theta(theta, angles, bell_masses, velocities, target_periods, verbose=False):
    global mode

    #angles = np.linspace(-np.pi-0.1, np.pi+0.1, 11)

    total_fitness = 0.0
    all_fitnesses = []
    for ai, init_angle in enumerate(angles):

        wheel_force = 600  # Max. force on the rope (in Newtons)
        count = 0
        fitness = 0.0

        ring_up = False
        ring_down = False
        ring_steady = False
        ring_up_back = False

        Net_local = ForceNet(n_nodes, n_inputs)

        Net_local.update_network(theta)

        sim = run_bell()

        sim.bell.current_mode = mode

        sim.bell.bell_angle = init_angle

        sim.bell.clapper_angle = np.sign(sim.bell.bell_angle)*sim.bell.clapper_limit + sim.bell.bell_angle

        sim.bell.stay_break_limit = 0.25

        sim.bell.velocity = velocities[ai]

        sim.bell.m_1 = bell_masses[ai]
        sim.bell.m_2 = 0.05*sim.bell.m_1

        sim.bell.target_period = target_periods[ai]  #The ideal length handstroke-handstroke(or vice versa)

        sim.bell.strict_rhythm = True   #This fixes the whole thing to the rhythm set by the first handstroke

        if np.abs(sim.bell.bell_angle) < 0.5:
            sim.bell.max_length = 0.0  # max backstroke length
        else:
            sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)

        # Run the given simulation for up to num_steps time steps.

        while sim.phy.time < max_time and sim.bell.strike_count < 52:
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

            if sim.bell.current_mode == 'up_back':
                ring_up_back = True
                action = Net_local.force(inputs)
                force = min(1.0, action[0])

            sim.bell.pull = force
            sim.step(force)

            #fitness = fitness + sim.bell.fitness_increment(sim.phy)

            sim.phy.count = sim.phy.count + 1

            if (sim.bell.stay_touch > 0 and sim.bell.bell_angle < np.pi) or sim.bell.stay_hit > 0:
                break

        fitness = sim.bell.fitness_fn(sim.phy, verbose=verbose)

        # if sim.bell.stay_hit > 0:
        #     sim.bell.stay_angle = 1e6
        #     fitness = 1.0

        total_fitness += fitness
        all_fitnesses.append(fitness)



    return sim.bell.handstroke_accuracy, sim.bell.backstroke_accuracy

max_time = 120.0
mode = 'steady'
load_best = True
extend_net = True
Net = ForceNet(n_nodes, n_inputs)
counter = 0  #Start at this one
plotcount = 0

while True:

    #Want to get this to automatically plot the best one
    fname = f'./nets/{mode}.txt'

    scores = []; best_scores = []
    #Determine the correct number of parameters for this best state
    if os.path.exists(fname):
        with open(fname, "r") as f:
            data = f.readlines()

    if len(data) > counter:
        print('Evaluating generation', counter)
        Net.load_specific_state(mode, counter)

        fig = plt.figure(figsize = (2.5,10))
        #Do a thing here to draw out the 'rounds'. Only need the bell strike times really. Handstroke should always be first

        selected_bell = 4  #Everyone else is perfect, just this bell isn't. If it ever gets any good, can change this
        nbells = 8
        nstrikes = 52

        for bell in range(nbells):
            if bell != selected_bell:
                plt.plot((bell+1)*np.ones(nstrikes), np.arange(nstrikes) + 1, c = 'black', linewidth=0.5)
            else:
                plt.plot((bell+1)*np.ones(nstrikes), np.arange(nstrikes) + 1, c = 'black', linewidth=0.25)

        #Data exists, go for it
        latest = data[counter].split(' ')
        generation = float(latest[0])
        score = float(latest[1])
        nnodes_actual = int(float(latest[3]))

        # masses = [100,200,300,400,500]
        # periods = [3.5,4.0,4.5,5.0,5.5]

        sampler = hpc(d=2)
        samples = sampler.random(50)

        masses = (samples[:,0]*10) + 295
        periods = (samples[:,1]*0.1) + 5.0

        for i, mass in enumerate(masses):
            target_period = periods[i]
            bell_cadence = target_period/(nbells*2 + 1)   #Distance between each bell

            handstroke_accuracy, backstroke_accuracy = evaluate_theta(Net.parameter_set, [np.pi-0.1], [mass], [0], [target_period])
            nstrokes_full = min(len(handstroke_accuracy[:]), len(backstroke_accuracy[:]))

            #Find accuracy position on chart
            strike_pos = []
            handstroke_accuracy[0] = 0.0
            for stroke in range(nstrokes_full):
                strike_pos.append(-handstroke_accuracy[stroke]/bell_cadence + selected_bell + 1)
                strike_pos.append(-backstroke_accuracy[stroke]/bell_cadence + selected_bell + 1)

            plt.plot(strike_pos, np.arange(2*nstrokes_full) + 1, c = 'red', linewidth=0.75)

        plt.xlim(-1,nbells+2)
        plt.ylim(nstrikes+1, 0)
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])
        plt.title(f'Generation {int(generation)}')
        #plt.axis('equal')
        plt.tight_layout()

        plt.savefig('./plots/methodplots/rounds_%05d.png' % plotcount)
        #plt.show()
        plt.close()

        counter += 10
        plotcount += 1
        print('Completed and plots saved')

    else:
        time.sleep(5.0)






