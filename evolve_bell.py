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

runs_per_net = 30
simulation_seconds = 60.0
ngenerations = 2000
up_time = 0.0   #Only measure performance after this point

use_existing_population = True #Load an existing network that is presumably better than nothing
use_best = True #Use best population rather than the most recent one

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

        if False:
            if random.random() < 0.0:   #pick a random angle
                sim.bell.bell_angle = uniform(-np.pi-sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
                sim.bell.clapper_angle = sim.bell.bell_angle
            else:
                if random.random() < 0.5:   #important that it can get itself off at hand and back
                    sim.bell.bell_angle = uniform(np.pi+0.95*sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
                    sim.bell.clapper_angle = sim.bell.bell_angle + sim.bell.clapper_limit - 0.01
                else:
                    sim.bell.bell_angle = uniform(-np.pi-0.95*sim.bell.stay_angle, -np.pi-sim.bell.stay_angle)
                    sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

        #amin = -np.pi-sim.bell.stay_angle; amax = np.pi+sim.bell.stay_angle   #range of initial conditions. Need a bit of randomness
        #Now some symmetry
        amin = 0.0*np.pi; amax = 0.1*np.pi
        rmin = (runs//2)*(amax - amin)/(runs_per_net//2) + amin
        rmax = (runs//2+1)*(amax - amin)/(runs_per_net//2) + amin

        sim.bell.bell_angle = uniform(rmin, rmax)

        if runs%2 == 0:
            sim.bell.bell_angle = -sim.bell.bell_angle

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
            # Inputs are the things we can know -- in my case it is the angle and speed of the bell (for now)
            # Do try to remember to get inputs in the range (0,1). Can do easily enough.
            inputs = sim.get_scaled_state()[:2]
            # This is just a list.
            action = net.activate(inputs)
            # Apply action to the simulated cart-pole
            force = continuous_actuator_force(action)
            sim.bell.pull = force
            sim.step(force)

            strike_limit = 1.0#5 seconds out in each direction to begin with
            sim.bell.strike_limit = strike_limit

            if sim.phy.time > up_time:
                fitness = fitness + sim.bell.fitness_increment(sim.phy)*(simulation_seconds)/(simulation_seconds - up_time)
        fitness = sim.bell.fitness_fn(sim.phy)

        fitnesses.append(fitness)
    # The genome's fitness is now its average.
    avg = sum(fitnesses)/len(fitnesses)
    #if max(fitnesses) > 1:
    #    print(avg, min(fitnesses))
    #if avg > 1:
    #    print('FITNESS', fitnesses)

    if min(fitnesses) < 1.0:
        return min(fitnesses)
    else:
        return avg
    #return sum(fitnesses)/len(fitnesses)


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
        if not use_best:
            with open("population_data/population", "rb") as f:
                pop_old = pickle.load(f)
            with open("population_data/species", "rb") as f:
                species_old = pickle.load(f)
            with open("population_data/generation", "rb") as f:
                generation_old = pickle.load(f)
        else:
            with open("population_data/population_best", "rb") as f:
                pop_old = pickle.load(f)
            with open("population_data/species_best", "rb") as f:
                species_old = pickle.load(f)
            with open("population_data/generation_best", "rb") as f:
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
    #pe = neat.ParallelEvaluator(1, eval_genome)
    winner = pop.run(pe.evaluate, n=ngenerations)

    # Save the winner.
    with open("winner_bell", "wb") as f:
        pickle.dump(winner, f)


if __name__ == "__main__":
    run()
