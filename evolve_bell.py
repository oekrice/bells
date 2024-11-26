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

runs_per_net = 25
simulation_seconds = 60.0
ngenerations = 500

use_existing_population = True #Load an existing network that is presumably better than nothing

if not use_existing_population:
    for i in range(0,10000):
        if os.path.isfile('./current_network/%d' % i):
            os.remove('./current_network/%d' % i)

# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = run_bell()  # all the physics in here

        if True:   #Conditions for general ringing. Start stood at each stroke I think, but might change that.
            if True:#runs < int(runs_per_net/2):
                sim.bell.bell_angle = uniform(np.pi+0.5*sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
                sim.bell.clapper_angle = sim.bell.bell_angle + sim.bell.clapper_limit - 0.01
            else:
                sim.bell.bell_angle = uniform(-np.pi-0.5*sim.bell.stay_angle, -np.pi -sim.bell.stay_angle)
                sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

            sim.bell.target_period = uniform(5.9,6.1)

            sim.bell.m_1 = uniform(390,410)
            sim.bell.m_2 = 0.05*sim.bell.m_1
            sim.bell.stay_break_limit = 0.4

        if np.abs(sim.bell.bell_angle) < 0.5:
            sim.bell.max_length = 0.0  # max backstroke length
        else:
            sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.phy.time < simulation_seconds:
            # Inputs are the things we can know -- in my case it is the angle and speed of the bell (for now)
            # Do try to remember to get inputs in the range (0,1). Can do easily enough.
            inputs = sim.get_scaled_state()
            # This is just a list.
            action = net.activate(inputs)
            # Apply action to the simulated cart-pole
            force = continuous_actuator_force(action)
            sim.bell.pull = force
            sim.step(force)

            strike_limit = 0.85#5 seconds out in each direction to begin with
            #Exit if out of bounds
            if len(sim.bell.handstroke_accuracy) > 1:
                if np.abs(sim.bell.handstroke_accuracy[-1]) > strike_limit:
                    break
                if len(sim.bell.backstroke_accuracy) > 1:
                    if np.abs(sim.bell.backstroke_accuracy[-1]) > strike_limit:
                        break
                if sim.bell.stay_hit > 0:
                    break

        fitness = sim.bell.fitness_fn(sim.phy)

        fitnesses.append(fitness)
    # The genome's fitness is now its average.
    #return min(fitnesses)
    return np.sum(fitnesses)/len(fitnesses)


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

    if use_existing_population:
        with open("population_data/population", "rb") as f:
            pop_old = pickle.load(f)
        with open("population_data/species", "rb") as f:
            species_old = pickle.load(f)
        with open("population_data/generation", "rb") as f:
            generation_old = pickle.load(f)
        initial_state = (pop_old, species_old, generation_old)
        pop = neat.Population(config, initial_state = initial_state)
    else:
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
