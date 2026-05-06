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
    bell.stay_break_limit = 0.5# 0.15 seems to be the limit of dropping the bell

    bell.m_1 = 500   #Bell mass
    bell.m_2 = 0.05*bell.m_1   #Clapper mass

    return bell

n_nodes = 2
n_inputs = 6
Net = ForceNet(n_nodes, n_inputs)
Net.generate_random_seed()

#nets = Networks()  #This is the old networks one

strike_limit = 1.0

simulation_seconds = 60.0
mode = 'up'
load_best = False
extend_net = True


if extend_net:
    n_nodes_target = n_nodes
    print(f'Extending net to {n_nodes_target} nodes')
    Net.extend_net(n_nodes_target=n_nodes_target)
    n_nodes = n_nodes_target

def evaluate_theta(theta, angles):
    global mode

    total_fitness = 0.0

    for initial_angle in angles:
        #Perhaps start off near the top and gradually work down? Worth a shot.

        # phy = init_physics()
        # phy.do_volume = False

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

        sim.bell.bell_angle = initial_angle

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
            force = 0.0  # This value between 0 and 1 and then update based on the physics.

            inputs = sim.bell.get_scaled_state()

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

            #force = 1.0/(1.0 + np.exp(-10*inputs[0]))
            sim.bell.pull = force#1.0/(1.0 + np.exp(-x))

            sim.step(force)

            fitness = fitness + sim.bell.fitness_increment(sim.phy)

            sim.phy.count = sim.phy.count + 1

        # Check for force on wheel - this takes effect at the next timestep
        # Check for actions or stay smash. All needs to be in the same event.get for some reason.
        fitness = (np.pi + sim.bell.stay_angle - np.max(sim.bell.bell_angles))**2

        if sim.bell.stay_hit > 0:
            sim.bell.stay_angle = 1e6
            fitness = (2*sim.bell.stay_angle)**2  #Stay break penalty (quite extreme)
            #print('Stay broken')
            break

        count += 1
        print('Max. angle', np.max(sim.bell.bell_angles), 'fitness', fitness)

        #print(fitness, phy.time, bell.bell_angle, bell.velocity)
        total_fitness += fitness

        print('Fitness for angle', initial_angle, ':', fitness)

    if total_fitness > 1e6:
        raise Exception("Run unsuccessful")
        total_fitness = 1e9

    return total_fitness

if load_best:
    Net.load_best_state(mode, override_nnodes=True)
    #Net.load_latest_state(mode)
    print('Loaded best state')
else:
    Net.generate_random_seed()
    print('Generated random state')

if extend_net:
    print(f'Extending net to {n_nodes_target} nodes')
    Net.extend_net(n_nodes_target=n_nodes_target)
    n_nodes = n_nodes_target

#fitness = evaluate_theta(Net.parameter_set)

def safe_evaluate_theta(theta, angles, sigma):
    for _ in range(3):
        try:
            return evaluate_theta(theta, angles)
        except:
            theta = theta + 0.1 * sigma * np.random.randn(*theta.shape)

    return 1e9

def run_cma_mp(n_cores=None):
    global mode
    global initial_angle
    if n_cores is None:
        n_cores = 1

    pool = mp.Pool(processes=n_cores)
    best_loss = float("inf")
    best_theta = None

    nsamples = 1
    popsize = n_cores
    while popsize < 32:
         popsize += n_cores

    print('Ncores:', n_cores, 'Population size', popsize)

    es = cma.CMAEvolutionStrategy(Net.parameter_set, 0.1, {'verb_disp': 1, 'popsize': popsize})

    with mp.Pool(processes=n_cores) as pool:

        while not es.stop():

            #Let's try learning from near the top. See what that comes out with.
            angles = np.random.uniform(0.0*np.pi, np.pi+0.1, nsamples)

            angles = angles*np.sign(np.random.uniform(-1,1,nsamples))

            angles = [0.5*np.pi]

            print('Initial angles:', angles)
            solutions = es.ask()

            results = [
                pool.apply_async(safe_evaluate_theta, (theta, angles, es.sigma))
                for theta in solutions
            ]

            losses = []
            for r in results:
                try:
                    losses.append(r.get(timeout=10.0))
                except Exception:
                    print('Run failed for some reason...')
                    losses.append(1e9)

            es.tell(solutions, losses)

            for theta, loss in zip(solutions, losses):
                #print("Current score", loss)

                if loss == np.min(losses):
                    best_theta_local = theta

            #Evaluate from zero to see if it's actually getting any better...

            angles = [0.0]
            Net.update_network(best_theta_local)

            #loss = safe_evaluate_theta(best_theta_local, angles, es.sigma)
            loss = evaluate_theta(best_theta_local, angles)

            Net.save_current_state(mode, loss)

            print('Actual loss for this generation (from completely down):', loss)
            if loss < best_loss:
                best_loss = loss
                best_theta = best_theta_local.copy()

            print(es.countiter)

    return

#run_cma_mp(n_cores=8)

evaluate_theta(Net.parameter_set, [0.5*np.pi])
#evaluate_theta(Net.parameter_set)




