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


#nets = Networks()  #This is the old networks one

strike_limit = 1.0

max_time = 30.0
mode = 'steady'


extend_net = True

#Let's have a look at just seeing whether the highest point increases after a couple of swings. Only need 15 seconds or so?

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

    alpha = 4

    all_fitnesses = np.array(all_fitnesses)
    total_fitness = (np.sum(all_fitnesses**alpha)/len(angles))**(1.0/alpha)
    #total_fitness = total_fitness/len(angles)
    #print('Total fitness', total_fitness)
    if total_fitness > 1e6:
        total_fitness = 1e12

    return total_fitness


#fitness = evaluate_theta(Net.parameter_set)
all_sigmas = []
initial_sigma = 2.5
terminate_sigma = 1.25

def run_cma_mp(n_nodes, n_inputs, n_cores=None):
    global mode
    global all_sigmas
    global initial_sigma
    global terminate_sigma
    if n_cores is None:
        n_cores = 1

    Net = ForceNet(n_nodes, n_inputs)

    if os.path.exists(f'./nets/{mode}.txt'):
        load_best = True
    else:
        load_best = False

    if load_best:
        do_latest = False
        Net.load_best_state(mode, override_nnodes=extend_net, latest=do_latest)
        print('Loaded set', Net.parameter_set)
        if do_latest:
            print('Loaded latest state')
        else:
            print('Loaded best state')
    else:
        Net.generate_random_seed()
        print('Generated random state')

    try:
        local_sigma_log = np.loadtxt('./nets/sigmas_nnodes.txt', delimiter = ',')
        max_prev_sigma = np.max(local_sigma_log)
        initial_sigma = 0.9*max_prev_sigma
        terminate_sigma = 0.5*max_prev_sigma
        print('Loaded sigma log. Previous maximum is:', max_prev_sigma)
        fname = f'./nets/{mode}.txt'
        #Determine the correct number of parameters for this best state
        best_score = 1.0
        if os.path.exists(fname):
            with open(fname, "r") as f:
                data = f.readlines()
                if len(data) > 2:
                    print('Reading score data...')
                    cut = min(10, len(data) - 1)
                    for li, line in enumerate(data[-cut:]):
                        if float(line.split(' ')[1]) < best_score:
                            best_score = float(line.split(' ')[1])
                else:
                    print('Not enough score data, sticking with 1.0 as threshold')
                    best_score = 1.0
        print('Loaded net log. Previous minimum is:', best_score)
        quality_threshold = best_score
    except:
        initial_sigma = 2.5  #This will change with each generation
        terminate_sigma = 1.25
        quality_threshold = 1.0

    # # -- THIS NEEDS TO BE REMOVED LATER
    # if n_nodes == 18:
    #     quality_threshold = 1.0

    #Establish a quality threshold
    nnodes_sigmas = []
    pool = mp.Pool(processes=n_cores)
    best_loss = float("inf")
    best_theta = None

    popsize = n_cores
    while popsize < 32:
         popsize += n_cores

    print('Ncores:', n_cores, 'Population size', popsize)

    print('Running new optimisation with parameters', n_nodes, n_inputs, initial_sigma, terminate_sigma)

    es_go = True
    es = cma.CMAEvolutionStrategy(Net.parameter_set, initial_sigma, {'verb_disp': 1, 'popsize': popsize})

    Net_best = ForceNet(n_nodes, n_inputs)
    local_losses = []

    with mp.Pool(processes=n_cores) as pool:
        while not es.stop() and es_go:

            #Set up slightly random distribution of angles

            n_interiors = 50
            width = 2*np.pi/n_interiors
            end_angles = [-np.pi-0.1 + np.random.uniform(-0.025,0.025), np.pi+0.1 + np.random.uniform(-0.025,0.025)]
            interior_angles = np.linspace(-0.9*np.pi, 0.9*np.pi, n_interiors) + np.random.uniform(-0.1,0.1, n_interiors).tolist()

            end_height = np.pi+0.15
            n_angles = 51
            # angles = np.linspace(-end_height,end_height,n_angles)
            # angles += np.random.uniform(-0.025,0.025, n_angles)


            angle_ends = np.linspace(-end_height,end_height,n_angles+1)
            #angle_ends = np.linspace(0.9*np.pi,np.pi+0.15)

            angles = np.random.uniform(angle_ends[:-1], angle_ends[1:])
            #angles*= np.random.choice([-1,1], len(angles))
            #angles = [0.0]
            #Now going to put some of the randomness in the mass rather than the angles. Can combine both eventually.
            #interior_angles = [-np.pi+0.1 + np.random.uniform(-0.025,0.025), np.pi-0.1 + np.random.uniform(-0.025,0.025)]
            #angles =  interior_angles

            #angles = [-np.pi-0.1, np.pi+0.1, 0.0]
            #angles = [np.random.uniform(-0.1,0.1)]
            #angles = [0.0]

            if False: #For down training
                velocities = np.random.uniform(-0.0,0.0,len(angles))
                bell_masses = np.random.uniform(100,500,len(angles))

            #Below for steady training:
            angles = np.random.uniform(np.pi*0.75,np.pi+0.1, n_angles)
            angles*= np.random.choice([-1,1], len(angles))
            angles += np.random.uniform(-0.025,0.025, n_angles)

            velocities = np.random.uniform(-0.0,0.0,len(angles))
            bell_masses = np.random.uniform(100,500,len(angles))
            target_periods = np.random.uniform(3.0,6.0,len(angles))
            #bell_masses = np.random.choice([500], size=len(angles))  #Just do the extremes

            print('Bell mass range:', np.min(bell_masses), np.max(bell_masses))
            print('Sample angle(s):', angles)

            solutions = es.ask()

            results = [
                pool.apply_async(evaluate_theta, (theta,angles,bell_masses,velocities,target_periods))
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

            #angles = [0.0]
            #Net.update_network(best_theta_local)

            #loss = safe_evaluate_theta(best_theta_local, angles, es.sigma)
            #loss = evaluate_theta(best_theta_local, angles, 600*np.ones(len(angles)), verbose=True)  #Evaluate with the heaviest the bell can be. Should be the worst performance.

            Net_best.update_network(best_theta_local)
            Net_best.save_current_state(mode, np.min(losses))

            local_losses.append(np.min(losses))

            print('Average loss:', np.mean(losses))
            print('Worst loss:', np.max(losses))
            print('Best loss for this generation:', np.min(losses))

            #print('Loss for m1 = 500:', loss)
            print(f'Count: {es.countiter}, sigma: {es.sigma}/{terminate_sigma}, nnodes:{n_nodes}, loss:{np.min(local_losses)}/{quality_threshold}:')
            all_sigmas.append(es.sigma)
            nnodes_sigmas.append(es.sigma)
            np.savetxt('./nets/sigmas_all.txt', all_sigmas, delimiter = ',')
            np.savetxt('./nets/sigmas_nnodes.txt', nnodes_sigmas, delimiter = ',')

            if es.sigma < terminate_sigma and np.min(local_losses) < 1.0*quality_threshold:
                es_go = False

            terminate_sigma = 0.5*np.max(nnodes_sigmas)

    return


if not test_mode:
    n_nodes = 6
    n_inputs = 13

    while n_nodes < 100:
        #Do the entire run
        run_cma_mp(n_nodes, n_inputs, n_cores=8)
        n_nodes += 2

else:
    max_time = 60.0
    n_nodes = 6
    n_inputs = 13
    Net = ForceNet(n_nodes, n_inputs)
    Net.load_best_state(mode, override_nnodes=extend_net, latest=True)

    for angle in np.linspace(np.pi-0.2, np.pi-0.1, 12):
        print('Angle:', angle)
        fitness = evaluate_theta(Net.parameter_set, [angle], [500], np.array([0]), [4.0], verbose = True)




