"""
Single-pole balancing experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle

from learn import run_bell, discrete_actuator_force, continuous_actuator_force, probably_actuator_force

import learn
import neat
import numpy as np
from random import uniform, gauss
import random

runs_per_net = 5
simulation_seconds = 60.0
ngenerations = 1000


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = run_bell()  # all the physics in here
        if random.random() < 0.75:
            sim.bell.bell_angle = uniform(np.pi+sim.bell.stay_angle-0.01, np.pi-sim.bell.stay_angle+0.01)
        else:
            sim.bell.bell_angle = uniform(-0.01,0.01)

        sim.bell.velocity = 0.0#uniform(-2.5,2.5)
        # Run the given simulation for up to num_steps time steps.

        while sim.phy.time < simulation_seconds:
            # Inputs are the things we can know -- in my case it is the angle and speed of the bell (for now)
            # Do try to remember to get inputs in the range (0,1). Can do easily enough.
            inputs = sim.get_scaled_state()
            # This is just a list.
            action = net.activate(inputs)
            # Apply action to the simulated cart-pole
            force = continuous_actuator_force(action)

            sim.step(force)

        fitness = sim.bell.fitness_fn()
        fitnesses.append(fitness)
    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "networks/config")
    # Load in config file. Will tweak in due course.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

    pop = neat.Population(config)
    # These just print some things out. But keep on...
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count() - 1, eval_genome)

    winner = pop.run(pe.evaluate, n=ngenerations)

    # Save the winner.
    with open("winner_bell", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    run()
