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

successes = []
ngenomes = 1000
for genome_test_number in range(ngenomes):
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

        if check == 0:
            sim.bell.bell_angle = uniform(np.pi+1.0*sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
            sim.bell.clapper_angle = sim.bell.bell_angle + sim.bell.clapper_limit - 0.01
        else:
            sim.bell.bell_angle = uniform(-np.pi-1.0*sim.bell.stay_angle, -np.pi-sim.bell.stay_angle)
            sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

        sim.bell.velocity = 0.0


        fitness = 0
        while sim.phy.time < 60.0:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)
            force = continuous_actuator_force(action)
            sim.step(force)
            fitness = fitness + sim.bell.fitness_increment(sim.phy)

        genome_success += abs(sim.bell.bell_angle) + abs(sim.bell.velocity)

    print('Genome', genome_test_number, genome_success)
    successes.append(genome_success)

plt.plot(np.arange(len(successes)), successes)
plt.xlabel('Generation')
plt.ylabel('Downness')
plt.yscale('log')
plt.show()




















