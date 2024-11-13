"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle

import neat
import sys

# from cart_pole import CartPole, discrete_actuator_force
from learn import run_bell, discrete_actuator_force, continuous_actuator_force, probably_actuator_force

import random
from random import uniform, gauss

import matplotlib.pyplot as plt

import numpy as np

# from movie import make_movie

# load the winner

if len(sys.argv) > 1:
    load_num = int(sys.argv[1])
else:
    load_num = -1

if load_num < 0:
    with open("current_best", "rb") as f:
        c = pickle.load(f)

else:
    with open("./current_network/%d" % (load_num ), "rb") as f:
        c = pickle.load(f)

print("Loaded genome:")
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, "networks/config")
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

# c is the best loaded genome. Easy to miss that...

net = neat.nn.FeedForwardNetwork.create(c, config)
sim = run_bell()
# Run the given simulation for up to 120 seconds.

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

sim.bell.velocity = 0.0

if random.random() < 1.0:
    sim.bell.bell_angle = 0.0
    sim.bell.clapper_angle = 0.0
else:
    sim.bell.bell_angle = uniform(-np.pi-0.95*sim.bell.stay_angle, -np.pi-sim.bell.stay_angle)
    sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

angles_log = [sim.bell.bell_angle]
velocities_log = [sim.bell.velocity]

print()
print("Initial conditions:")
print("    angle = {0:.4f}".format(sim.bell.bell_angle))
print(" velocity = {0:.4f}".format(sim.bell.velocity))

fitness = 0
while sim.phy.time < 60.0:
    inputs = sim.get_scaled_state()
    action = net.activate(inputs)

    # Apply action to the simulated cart-pole
    #force = discrete_actuator_force(action)
    force = continuous_actuator_force(action)
    #force = probably_actuator_force(action)
    #print(force, inputs)
    # Doesn't matter what you do with this as long as it's consistent.
    sim.step(force)
    angles_log.append(sim.bell.bell_angle)
    velocities_log.append(sim.bell.velocity)
    fitness = fitness + sim.bell.fitness_increment(sim.phy)
    #print(sim.bell.fitness_increment(sim.phy)*60*60)

#fitness = sim.bell.fitness_fn()
print("fitness", fitness)

print()
print("Final conditions:")
print("    angle = {0:.4f}".format(sim.bell.bell_angle))
print(" velocity = {0:.4f}".format(sim.bell.velocity))
print()

plt.plot(sim.bell.times, angles_log)
plt.plot(sim.bell.times, 0.0*np.ones(len(sim.bell.times)),linestyle = 'dotted')
plt.plot(sim.bell.times, np.pi*np.ones(len(sim.bell.times)),linestyle = 'dashed')
plt.plot(sim.bell.times, -np.pi*np.ones(len(sim.bell.times)),linestyle = 'dashed')

plt.ylim(-np.pi-sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
plt.close()

maxvel = np.max(np.abs(velocities_log))

def plot_forces():
    #Does a colourmap of the forces based on the input states
    angles = np.linspace(-np.pi-sim.bell.stay_angle, np.pi+sim.bell.stay_angle, 250)
    velocities = np.linspace(-10.0, 10.0, 300)
    mat = np.zeros((len(angles), len(velocities)))
    for i, angle in enumerate(angles):
        for j, velocity in enumerate(velocities):
            mat[i,j] = net.activate([angle / (np.pi + sim.bell.stay_angle), velocity / (10.0)])[0]
    plt.xlabel('Bell angle')
    plt.ylabel('Bell velocity')
    im = plt.pcolormesh(angles, velocities, mat.T, vmin = 0.0, vmax = 1.0, cmap = 'plasma')
    plt.contour(angles, velocities, mat.T, np.linspace(0.0,1.0,11), colors = 'black')
    plt.colorbar(im, label = 'Force')
    plt.plot(angles_log, velocities_log, c= 'white')
    plt.title('Generation %d, Fitness = %.3f' % (load_num, c.fitness))
    plt.savefig('network_graphs/%04d.png' % load_num)
    plt.close()

plot_forces()



















