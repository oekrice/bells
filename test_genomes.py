"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle

import neat

# from cart_pole import CartPole, discrete_actuator_force
from learn import run_bell, discrete_actuator_force, continuous_actuator_force, probably_actuator_force

import random
from random import uniform, gauss

import matplotlib.pyplot as plt

import numpy as np

successes = []; genomes = []
ngenomes = 1000
max_time = 120.0

for genome_test_number in range(0,1000,1):
    load_num = genome_test_number

    if os.path.isfile("./current_network/%d" % (load_num )):

        with open("./current_network/%d" % (load_num ), "rb") as f:
            c = pickle.load(f)
    else:
        break
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "networks/config")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    # c is the best loaded genome. Easy to miss that...

    net = neat.nn.FeedForwardNetwork.create(c, config)

    genome_success = 0.

    for check in range(2):
        sim = run_bell()

        if True:
            if check == 0:
                sim.bell.bell_angle = uniform(np.pi+1.0*sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
                sim.bell.clapper_angle = sim.bell.bell_angle + sim.bell.clapper_limit - 0.01
            else:
                sim.bell.bell_angle = uniform(-np.pi-1.0*sim.bell.stay_angle, -np.pi-sim.bell.stay_angle)
                sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

            sim.bell.velocity = 0.0
        else:
            if check == 0:
                sim.bell.bell_angle = 0.0
                sim.bell.clapper_angle = 0.0
            else:
                sim.bell.bell_angle = uniform(-np.pi-1.0*sim.bell.stay_angle, -np.pi-sim.bell.stay_angle)
                sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

        if np.abs(sim.bell.bell_angle) < 0.5:
            sim.bell.max_length = 0.0  # max backstroke length
        else:
            sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)


        fitness = 0

        while sim.phy.time < max_time:
            inputs = sim.get_scaled_state()[:2]
            action = net.activate(inputs)
            force = continuous_actuator_force(action)
            sim.bell.pull = force
            sim.step(force)
            fitness = fitness + sim.bell.fitness_increment(sim.phy)

        if True:   #Down
            genome_success += abs(sim.bell.bell_angle) + abs(sim.bell.velocity)
        else:   #Up
            genome_success = up_time

    fitness = sim.bell.fitness_fn(sim.phy, print_accuracy = True)
    print('Genome', genome_test_number, fitness, sim.bell.bell_angle, sim.bell.velocity)
    successes.append(fitness)
    genomes.append(genome_test_number)

plt.plot(genomes, successes)
plt.xlabel('Generation')
plt.ylabel('Downness')
plt.savefig('down2.png')

plt.show()




















