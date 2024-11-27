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

sim.bell.m_1 = uniform(150,550)
sim.bell.m_2 = 0.05*sim.bell.m_1
if random.random() < 1.0:
    sim.bell.bell_angle = uniform(np.pi+0.5*sim.bell.stay_angle, np.pi+sim.bell.stay_angle)
    sim.bell.clapper_angle = sim.bell.bell_angle + sim.bell.clapper_limit - 0.01
else:
    sim.bell.bell_angle = uniform(-np.pi-0.5*sim.bell.stay_angle, -np.pi -sim.bell.stay_angle)
    sim.bell.clapper_angle = sim.bell.bell_angle - sim.bell.clapper_limit + 0.01

#sim.bell.bell_angle = np.pi-0.5
#sim.bell.target_period = uniform(4.0,5.4)

if np.abs(sim.bell.bell_angle) < 0.5:
    sim.bell.max_length = 0.0  # max backstroke length
else:
    sim.bell.max_length = sim.bell.radius*(1.0 + 3*np.pi/2 - sim.bell.garter_hole)

sim.bell.target_period = 6.0

sim.bell.m_1 = 400
sim.bell.m_2 = 0.05*sim.bell.m_1

sim.bell.stay_break_limit = 0.4

print('Target period', sim.bell.target_period )
angles_log = [sim.bell.bell_angle]
velocities_log = [sim.bell.velocity]

print()
print("Initial conditions:")
print("    angle = {0:.4f}".format(sim.bell.bell_angle))
print(" velocity = {0:.4f}".format(sim.bell.velocity))
print(" bell mass = {0:.4f}".format(sim.bell.m_1))

fitness = 0
strike_limit = 5.0

while sim.phy.time < 60*sim.bell.target_period:
    inputs = sim.get_scaled_state()
    action = net.activate(inputs)

    # Apply action to the simulated cart-pole
    #force = discrete_actuator_force(action)
    force = continuous_actuator_force(action)
    #force = probably_actuator_force(action)
    #print(force, inputs)
    # Doesn't matter what you do with this as long as it's consistent.
    sim.bell.pull = force
    sim.step(force)
    angles_log.append(sim.bell.bell_angle)
    velocities_log.append(sim.bell.velocity)
    fitness = fitness + sim.bell.fitness_increment(sim.phy)
    #print(sim.bell.fitness_increment(sim.phy)*60*60)

    if len(sim.bell.handstroke_accuracy) > 1:
        #if np.abs(sim.bell.handstroke_accuracy[-1]) > strike_limit:
        #    break
        #if len(sim.bell.backstroke_accuracy) > 1:
        #    if np.abs(sim.bell.backstroke_accuracy[-1]) > strike_limit:
        #        break
        if sim.bell.stay_hit > 0:
            break

print(sim.bell.handstroke_accuracy)
print(sim.bell.backstroke_accuracy)
#print(sim.bell.forces)
fitness = sim.bell.fitness_fn(sim.phy, print_accuracy = True)
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
    angles = np.linspace(-np.pi-0.15, np.pi+0.15, 250)
    velocities = np.linspace(-10.0, 10.0, 300)
    hs = []
    for i in range(len(sim.bell.all_handstrokes)-1):
        hs.append(int(sim.bell.all_handstrokes[i+1]*sim.phy.FPS))
    bs = []
    for i in range(len(sim.bell.all_backstrokes)-1):
        bs.append(int(sim.bell.all_backstrokes[i+1]*sim.phy.FPS))

    #mat = np.zeros((len(angles), len(velocities)))
    #for i, angle in enumerate(angles):
    #    for j, velocity in enumerate(velocities):
    #        mat[i,j] = net.activate([angle / (np.pi + sim.bell.stay_angle), velocity / (10.0),sim.bell.m_1/1000])[0]
    plt.xlabel('Bell angle')
    plt.ylabel('Bell velocity')
    #im = plt.pcolormesh(angles, velocities, mat.T, vmin = 0.0, vmax = 1.0, cmap = 'plasma')
    #plt.contour(angles, velocities, mat.T, np.linspace(0.0,1.0,11), colors = 'black')
    #plt.colorbar(im, label = 'Force')
    plt.xlim(-np.pi-0.15, np.pi+0.15)
    plt.plot(angles_log, velocities_log, c= 'black')

    plt.scatter(np.array(angles_log)[np.array(hs, dtype='int')], np.array(velocities_log)[np.array(hs, dtype='int')], c = 'green',zorder= 10)
    plt.scatter(np.array(angles_log)[np.array(bs, dtype='int')], np.array(velocities_log)[np.array(bs, dtype='int')], c = 'red', zorder= 10)

    plt.title('Generation %d, Fitness = %.3f' % (load_num, c.fitness))
    plt.savefig('network_graphs/%04d.png' % load_num)
    if load_num < 0:
        plt.show()
    plt.show()
    plt.close()

#plot_forces()

def plot_rounds():
    #Plots the bell's ability in plot_rounds
    fig = plt.figure(figsize = (2,8))
    try:

        nbells = sim.bell.nbells

        sim.bell.all_handstrokes = sim.bell.all_handstrokes[1:]
        sim.bell.all_backstrokes = sim.bell.all_backstrokes[1:]

        if sim.bell.all_backstrokes[0] < sim.bell.all_handstrokes[0]:
            sim.bell.all_backstrokes[:] = sim.bell.all_backstrokes[1:]
            sim.bell.backstroke_accuracy[:] = sim.bell.backstroke_accuracy[1:]

        nhands = len(sim.bell.all_handstrokes)
        nbacks = len(sim.bell.all_backstrokes)

        rows = np.arange(nhands+nbacks)
        alltimes = []
        for b in range(nbells):
            alltimes.append([])

        #Plot the base lines
        for row in range(nhands+nbacks):
            row_start = sim.bell.all_handstrokes[0] + row*sim.bell.target_period
            row_times = np.linspace(0, sim.bell.target_period/2, sim.bell.nbells+1)  #Adjusted for time

            for b in range(nbells):
                alltimes[b].append(row_times[b])  #handstrokes
        for b in range(nbells):
            plt.plot(alltimes[b], rows, c= 'black', linewidth = 0.5)

        bell_num = 4   #Number in the change (doesn't matter for now really apart from the plot

        #Plot the accuracies
        sim.bell.handstroke_accuracy[0] = 0.0
        plot_times = []
        for row2 in range(nbacks):
            plot_times.append(row_times[bell_num - 1] - sim.bell.handstroke_accuracy[row2])
            plot_times.append(row_times[bell_num - 1] - sim.bell.backstroke_accuracy[row2])
        if nhands > nbacks:
            plot_times.append(row_times[bell_num - 1] - sim.bell.handstroke_accuracy[-1])

        plt.plot(plot_times, rows[:len(plot_times)], c= 'red', linewidth = 2.0)
        plt.plot(alltimes[bell_num - 1], rows, c= 'green', linewidth = 2.0)
        plt.xlim(2*row_times[0] - row_times[1], row_times[-1])

    except:
        pass
    plt.ylim(-5,125)

    plt.gca().invert_yaxis()
   # plt.title('Stroke Time = %.1fs' % sim.bell.target_period)
    plt.title('Generation = %.d' % load_num)
    plt.tight_layout()
    plt.savefig('rounds_graphs/%04d.png' % load_num)
    if load_num < 0:
        plt.show()
    plt.show()
    plt.close()

plot_rounds()














