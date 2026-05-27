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

        sim.bell.target_period = target_periods[ai]

        sim.bell.strict_rhythm = True

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

    fig = plt.figure(figsize=(5,5))

    c = 'black'

    for hand in sim.bell.all_handstrokes[1:]:
        hi = np.searchsorted(sim.bell.times, hand)
        plt.scatter(sim.bell.times[hi], sim.bell.bell_angles[hi], c = 'green')
    for back in sim.bell.all_backstrokes[1:]:
        bi = np.searchsorted(sim.bell.times, back)
        plt.scatter(sim.bell.times[bi], sim.bell.bell_angles[bi], c = 'red')

    cut = len(sim.bell.bell_angles)
    plt.plot(sim.bell.times[:cut], sim.bell.bell_angles,linewidth=1.0,c=c, zorder=0)
    plt.ylim(-np.pi-0.2,np.pi+0.2)
    plt.xlabel('Time')
    plt.ylabel('Bell Angle')
    plt.title(f'Generation {int(generation)}, score: {score:03f}')
    plt.xlim(-2.5, 62.5)
    plt.tight_layout()
    plt.savefig('./plots/timeplots/timeplot_%05d.png' % generation)
    plt.close()

    fig = plt.figure(figsize=(5,5))

    c = 'black'

    for hand in sim.bell.all_handstrokes[1:]:
        hi = np.searchsorted(sim.bell.times, hand)
        plt.scatter(sim.bell.bell_angles[hi], sim.bell.velocities[hi], c = 'green')
    for back in sim.bell.all_backstrokes[1:]:
        bi = np.searchsorted(sim.bell.times, back)
        plt.scatter(sim.bell.bell_angles[bi], sim.bell.velocities[bi], c = 'red')

    cut = len(sim.bell.bell_angles)
    plt.plot(sim.bell.bell_angles,sim.bell.velocities[:cut],linewidth=1.0,c=c, zorder=0)
    plt.xlim(-np.pi-0.2,np.pi+0.2)
    plt.ylim(-10,10.0)
    plt.xlabel('Bell Angle')
    plt.ylabel('Bell Angular Velocity')
    plt.title(f'Generation {int(generation)}, score: {score:03f}')
    plt.tight_layout()
    plt.savefig('./plots/phaseplots/phaseplot_%05d.png' % generation)
    plt.close()

    fig = plt.figure(figsize = (3,7))
    #Do a thing here to draw out the 'rounds'. Only need the bell strike times really. Handstroke should always be first
    nstrokes_full = min(len(sim.bell.all_handstrokes[1:]), len(sim.bell.all_backstrokes[1:]))

    bell_cadence = target_periods[0]/(sim.bell.nbells*2 + 1)   #Distance between each bell
    selected_bell = 4  #Everyone else is perfect, just this bell isn't. If it ever gets any good, can change this
    for bell in range(sim.bell.nbells):
        if bell != selected_bell:
            plt.plot((bell+1)*np.ones(30), np.arange(30) + 1, c = 'black', linewidth=1.0)
        else:
            plt.plot((bell+1)*np.ones(30), np.arange(30) + 1, c = 'black', linewidth=0.225)

    #Find accuracy position on chart
    strike_pos = []
    sim.bell.handstroke_accuracy[0] = 0.0
    for stroke in range(nstrokes_full):
        strike_pos.append(-sim.bell.handstroke_accuracy[stroke]/bell_cadence + selected_bell + 1)
        strike_pos.append(-sim.bell.backstroke_accuracy[stroke]/bell_cadence + selected_bell + 1)

    plt.plot(strike_pos, np.arange(2*nstrokes_full) + 1, c = 'blue')
    plt.xlim(-1,sim.bell.nbells+2)
    plt.ylim(31, 0)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.title(f'Generation {int(generation)}')
    plt.tight_layout()

    plt.savefig('./plots/lineplots/lineplot_%05d.png' % generation)

    plt.close()

    alpha = 4

    all_fitnesses = np.array(all_fitnesses)
    total_fitness = (np.sum(all_fitnesses**alpha)/len(angles))**(1.0/alpha)
    #total_fitness = total_fitness/len(angles)
    #print('Total fitness', total_fitness)
    if total_fitness > 1e6:
        total_fitness = 1e12

    return total_fitness

max_time = 60.0
mode = 'steady'
load_best = True
extend_net = True
Net = ForceNet(n_nodes, n_inputs)
counter = 0  #Start at this one

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

        #Data exists, go for it
        latest = data[counter].split(' ')
        generation = float(latest[0])
        score = float(latest[1])
        nnodes_actual = int(float(latest[3]))

        fitness = evaluate_theta(Net.parameter_set, [np.pi-0.1], [300], [0], [4])
        counter += 1
        print('Completed and plots saved')

    else:
        time.sleep(5.0)






