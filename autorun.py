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

if True:
    nest_asyncio.apply()

test_mode = False
if len(sys.argv) > 1:
    if sys.argv[1] == 'test':
        test_mode = True

audio_enabled = False

phy = init_physics()
phy.do_volume = False

n_nodes = 10
n_inputs = 6
Net = ForceNet(n_nodes, n_inputs)

#nets = Networks()  #This is the old networks one

strike_limit = 1.0

max_time = 15.0
mode = 'up'

if os.path.exists(f'./nets/{mode}.txt'):
    load_best = True
else:
    load_best = False

extend_net = True

#Let's have a look at just seeing whether the highest point increases after a couple of swings. Only need 15 seconds or so?

def evaluate_theta(theta, angles, verbose=False):
    global mode

    #angles = np.linspace(-np.pi-0.1, np.pi+0.1, 11)

    total_fitness = 0.0

    for init_angle in angles:

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

        fitness = sim.bell.fitness_fn(sim.phy, verbose=verbose)

        # if sim.bell.stay_hit > 0:
        #     sim.bell.stay_angle = 1e6
        #     fitness = 1.0

        total_fitness += fitness
    total_fitness = total_fitness/len(angles)
    #print('Total fitness', total_fitness)
    if total_fitness > 1e6:
        total_fitness = 1e12

    return total_fitness


#fitness = evaluate_theta(Net.parameter_set)

def run_cma_mp(n_cores=None):
    global mode
    if n_cores is None:
        n_cores = 1

    pool = mp.Pool(processes=n_cores)
    best_loss = float("inf")
    best_theta = None

    popsize = n_cores
    while popsize < 32:
         popsize += n_cores

    print('Ncores:', n_cores, 'Population size', popsize)

    es = cma.CMAEvolutionStrategy(Net.parameter_set, 0.5, {'verb_disp': 1, 'popsize': popsize})

    Net_best = ForceNet(n_nodes, n_inputs)

    with mp.Pool(processes=n_cores) as pool:
        while not es.stop():

            #Set up slightly random distribution of angles


            n_interiors = 50
            width = 2*np.pi/n_interiors
            end_angles = [-np.pi-0.1 + np.random.uniform(-0.025,0.025), np.pi+0.1 + np.random.uniform(-0.025,0.025)]
            interior_angles = np.linspace(-0.9*np.pi, 0.9*np.pi, n_interiors) + np.random.uniform(-0.1,0.1, n_interiors).tolist()

            n_angles = 51
            angles = np.linspace(-np.pi-0.1,np.pi+0.1,n_angles)
            #interior_angles = [-np.pi+0.1 + np.random.uniform(-0.025,0.025), np.pi-0.1 + np.random.uniform(-0.025,0.025)]
            #angles =  interior_angles

            #angles = [-np.pi-0.1, np.pi+0.1, 0.0]
            #angles = [np.random.uniform(-0.1,0.1)]
            #angles = [0.0]

            print('Sample angle(s):', angles)

            solutions = es.ask()

            results = [
                pool.apply_async(evaluate_theta, (theta,angles))
                for theta in solutions
            ]

            losses = []
            for r in results:
                losses.append(r.get())

                # try:
                #     losses.append(r.get(timeout=n_interiors))
                # except Exception:
                #     print('Timeout?')
                #     losses.append(1e9)

            es.tell(solutions, losses)

            for theta, loss in zip(solutions, losses):
                print("Current score", loss)

                if loss == np.min(losses):
                    best_theta_local = theta

            #Evaluate from zero to see if it's actually getting any better...

            angles = [0.0]
            #Net.update_network(best_theta_local)

            #loss = safe_evaluate_theta(best_theta_local, angles, es.sigma)
            #loss = evaluate_theta(best_theta_local, angles, verbose=True)

            Net_best.update_network(best_theta_local)
            Net_best.save_current_state(mode, np.min(losses))

            print('Average loss:', np.mean(losses))
            print('Worst loss:', np.max(losses))
            print('Best loss for this generation:', np.min(losses))

            print(es.countiter)
    return

if load_best:
    do_latest = True
    Net.load_best_state(mode, override_nnodes=extend_net, latest=do_latest)
    print('Loaded set', Net.parameter_set)
    if do_latest:
        print('Loaded latest state')
    else:
        print('Loaded best state')
else:
    Net.generate_random_seed()
    print('Generated random state')

if not test_mode:
    run_cma_mp(n_cores=8)
else:
    for angle in np.linspace(-np.pi-0.1, np.pi+0.1, 20):
        print('Angle:', angle)
        fitness = evaluate_theta(Net.parameter_set, [angle])




