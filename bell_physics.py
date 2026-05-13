# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:29:08 2024

@author: eleph
"""
import numpy as np
import time
import pygame, sys
from pygame.locals import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


class init_physics:
    # define some physical parameters, like gravity etc.
    # use m/s as units I think. Need to convert that to pixel space, naturally
    # Also various plotting tools in here, because I can't think where else to put them
    def __init__(self):
        self.pixels_x = 384
        self.pixels_y = 384
        self.FPS = 60
        self.g = 9.8  # Gravitational acceleration
        self.x1 = 1.5  # width of domain in 'metres'
        self.y1 = self.x1 * self.pixels_y / self.pixels_x
        self.dt = 1.0 / self.FPS
        self.xscale = self.pixels_x / self.x1
        self.yscale = self.pixels_y / self.y1
        self.count = 0
        self.game_time = 0.0
        self.time_reference = time.time()
        self.real_time = time.time() - self.time_reference
        self.do_volume = True
        self.time = 0.0

    def rotate(self, image, angle):
        # Rotates bell image. Need to be very careful with angles.
        init_w, _ = image.get_size()
        size_correction = init_w * np.sqrt(2) * np.cos(angle % (np.pi / 2) - np.pi / 4)
        rot_image = pygame.transform.rotate(image, 180 * angle / np.pi)
        x_box = -(size_correction / 2) / self.xscale
        y_box = -(size_correction / 2) / self.yscale

        return rot_image, (x_box, y_box)

    def draw_point(self, surface, pt_x, pt_y, colour):
        # Places a point on the screen at the coordinates x and y
        pygame.draw.circle(surface, colour, self.pix(pt_x, -pt_y), 5, 0)

    def pix(self, x, y):
        # Converts x and y coordinates in physics space to pixel space.
        # Centre of the image is 0,0
        return self.pixels_x / 2 + x * self.xscale, self.pixels_y / 2 + y * self.yscale


class init_bell:
    # class for attributes of the bell itself, eg. speed and location

    def __init__(self, phy, init_angle):

        self.radius = 0.5  # radius of wheel (in m)
        self.garter_hole = np.pi / 4  # position of the garter hole relative to the stay

        self.onedge = False
        self.ding = False
        self.ding_reset = True
        self.ding_time = 0.0

        self.accel = 0.0  # angular acceleration in radians/s^2
        self.velocity = 0.0  # angular velocity in radians/s
        self.bell_angle = init_angle

        self.backstroke_pull = 1.0  # length of backstroke pull in metres

        self.prev_angle = init_angle  # previous maximum angle

        self.rlength, self.effect_force = self.ropelength()
        if np.abs(self.bell_angle) < 0.5:
            self.max_length = 0.0  # max backstroke length
        else:
            self.max_length = self.radius*(1.0 + 3*np.pi/2 - self.garter_hole)
        self.rlengths = []
        self.effect_forces = []

        self.volume = 0.0

        self.l_1 = 0.7 * self.radius  # distance from the bell pivot to the bell COM
        self.k_1 = 1.5  # coefficient in the bell moment of inertia (I_1 = k_1m_1l_1^2)
        self.m_1 = 500.0  # mass of bell (in kg)
        self.wheel_force = 0.0  # force on the bell wheel (as in, rope pull)
        self.stay_angle = 0.15  # how far over the top can the bell go (elastic collision)
        self.friction = 0.025  # friction parameter in arbitrary units

        self.clapper_accel = 0.0  # clapper angular acceleration
        self.clapper_velocity = 0.0  # clapper angular velocity
        self.clapper_angle = self.bell_angle  # Angle of clapper RELATIVE TO GRAVITY

        self.p = 0.1 * self.radius  # distance of pivot point from the centre of the bell
        self.l_2 = 0.65 * self.radius  # clapper length
        self.k_2 = 1.5  # coefficient in the clapper moment of intertia

        self.m_2 = 0.05 * self.m_1  # mass of clapper
        self.clapper_limit = 0.3  # maximum clapper angle  (will need tweaking)
        self.onedge = False  # True if the clapper is in contact with the bell
        self.strike_velocity = 0.0
        self.volume_ref = 0.0
        self.clapper_friction = 0.1 * self.friction
        self.stay_break_limit = 1.0

        self.bell_angles = []
        self.velocities = []
        self.forces = []
        self.times = [0.0]
        self.stay_hit = 0
        self.stay_touch = 0
        self.stay_touch_velocity = 0.0 #The first time the stay is hit
        self.pull = 0.0

        self.all_handstrokes = [-10]   #Timing of the respective strokes, in seconds.
        self.all_backstrokes = [-10]
        self.last_handstroke = 10   #time since last ring
        self.last_backstroke = 10
        self.target_period = 4   #Target time between sucessive strokes
        self.nbells = 8 #Rhythm for handstroke gaps
        self.handstroke_accuracy = []   #Difference in the target times overall (for the fitness fn)
        self.backstroke_accuracy = []
        self.handstroke_target = -10.0
        self.backstroke_target = -10.0
        self.update_rhythm = False

        self.next_handstroke = self.target_period
        self.next_backstroke = self.target_period/2

        self.strike_limit = 100

        self.current_mode = 'none'   #This is whether it's ringing up, down etc.
        self.possible_force = 0.0

    def timestep(self, phy):
        # Do the timestep here, using only bell.force, which comes either from an input or the machine
        # Update the physics here
        self.time_targets(phy)

        if not self.onedge:  # CLAPPER IS NOT RESTING ON THE EDGE OF THE BELL

            # Acceleration due to gravity
            num = -self.m_1 * phy.g * self.l_1 * np.sin(self.bell_angle) - self.m_2 * phy.g * self.p * np.sin(self.bell_angle)
            num = num - self.m_2 * self.p * self.l_2 * self.clapper_velocity**2 * np.sin(self.bell_angle - self.clapper_angle)
            den = self.m_1 * ((1.0 + self.k_2) * self.l_1**2) + self.m_2 * self.p**2
            den = den + self.m_2 * self.p * self.l_2 * np.cos(self.bell_angle - self.clapper_angle)

            self.accel = num / den
            # self.accel = (-phy.g*np.sin(self.bell_angle))/((1.0 + self.k_1)*self.l_1)
            # Acceleration on the wheel due to the pull
            self.accel = self.accel + (1 / self.m_1) * self.wheel_force * self.radius / ((1.0 + self.k_1) * self.l_1**2)
            # Friction (proportional to angular velocity. Increases with weight for now)
            self.accel = self.accel - self.velocity * self.friction

            # Velocity timestep (forward Euler)
            self.velocity = self.velocity + self.accel * phy.dt
            # extra friction so it actually stops at some point
            if abs(self.velocity) < 0.01 and self.wheel_force == 0.0 and self.bell_angle >= np.pi:
                self.velocity = 0.5 * self.velocity
            if abs(self.bell_angle) < 1e-4 and abs(self.velocity) < 0.01:
                self.velocity = 0.5 * self.velocity

            self.prev_angle = self.bell_angle
            self.bell_angle = self.bell_angle + self.velocity * phy.dt

            # check if stay has been hit, and bounce if so
            if self.bell_angle > np.pi + self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = 2 * np.pi + 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                self.stay_touch_velocity = abs(self.velocity)
                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity
            if self.bell_angle < -np.pi - self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = -2 * np.pi - 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                self.stay_touch_velocity = abs(self.velocity)

                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            # Update location of the clapper (using some physics which may well be dodgy)
            num = -phy.g * np.sin(self.clapper_angle) - self.p * (
                self.accel * np.cos(self.bell_angle - self.clapper_angle)
                - self.velocity**2 * np.sin(self.bell_angle - self.clapper_angle)
            )
            den = (1.0 + self.k_2) * self.l_2
            self.clapper_accel = num / den
            self.clapper_velocity = self.clapper_velocity + self.clapper_accel * phy.dt
            self.clapper_accel = self.clapper_accel - self.clapper_friction * (self.clapper_velocity - self.velocity)
            # self.clapper_velocity = 0.0

            # Update clapper angle
            self.clapper_angle = self.clapper_angle + self.clapper_velocity * phy.dt

        else:  # Clapper is on the edge of the bell
            # Need to do the same physics initially as if they are not attached, to check if they should still be.
            self.accel = (-phy.g * np.sin(self.bell_angle)) / ((1.0 + self.k_1) * self.l_1)
            # Acceleration on the wheel
            self.accel = self.accel + (1 / self.m_1) * self.wheel_force * self.radius / ((1.0 + self.k_1) * self.l_1**2)
            # Friction (proportional to angular velocity. Increases with weight for now)
            self.accel = self.accel - self.velocity * self.friction

            old_velocity = self.velocity
            old_angle = self.bell_angle
            # Velocity timestep (forward Euler)
            self.velocity = self.velocity + self.accel * phy.dt
            # extra friction so it actually stops at some point
            if abs(self.velocity) < 0.01 and self.wheel_force == 0.0 and self.bell_angle >= np.pi:
                self.velocity = 0.5 * self.velocity
            if abs(self.bell_angle) < 1e-4 and abs(self.velocity) < 0.01:
                self.velocity = 0.5 * self.velocity

            self.prev_angle = self.bell_angle
            self.bell_angle = self.bell_angle + self.velocity * phy.dt

            # check if stay has been hit, and bounce if so
            if self.bell_angle > np.pi + self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = 2 * np.pi + 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                self.stay_touch_velocity = abs(self.velocity)

                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            if self.bell_angle < -np.pi - self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = -2 * np.pi - 2 * self.stay_angle - self.bell_angle
                self.stay_touch = self.stay_touch + 1
                self.stay_touch_velocity = abs(self.velocity)

                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            # Check if clapper needs to leave the bell
            num = -phy.g * np.sin(self.clapper_angle) - self.p * (
                self.accel * np.cos(self.bell_angle - self.clapper_angle)
                - self.velocity**2 * np.sin(self.bell_angle - self.clapper_angle)
            )
            den = (1.0 + self.k_2) * self.l_2
            self.clapper_accel = num / den
            self.clapper_accel = self.clapper_accel - self.clapper_friction * (self.clapper_velocity - self.velocity)

            if abs(self.velocity) < 0.05 and self.wheel_force == 0.0:
                if abs(self.bell_angle + np.pi + self.stay_angle) < 0.01 or abs(self.bell_angle - np.pi - self.stay_angle) < 0.01:
                    self.velocity = 0.0
                    self.bell_angle = np.sign(self.bell_angle) * (np.pi + self.stay_angle)

            elif self.clapper_accel * self.clapper_velocity > self.accel * self.velocity:
                # Do leave the bell, so update the clapper accordingly.
                # Bell acceleration at this point is fine
                self.onedge = False
                # update (but no friction initially)
                self.clapper_velocity = self.clapper_velocity + self.clapper_accel * phy.dt
                # Update clapper angle
                self.clapper_angle = self.clapper_angle + self.clapper_velocity * phy.dt

            else:
                # Clapper should still be attached, so scrap that physics and treat it as one body
                num = -self.l_1 * self.m_1 * phy.g * np.sin(old_angle) - self.m_2 * phy.g * (
                    self.p * np.sin(old_angle) + self.l_2 * np.sin(self.clapper_angle)
                )
                den = self.m_1 * ((1.0 + self.k_1) * self.l_1**2) + self.m_2 * (
                    (1.0 + self.k_2) * (self.p + self.l_2 * np.cos(old_angle - self.clapper_angle)) ** 2
                )
                self.accel = num / den

                # Acceleration on the wheel (this isn't quite accurate but meh)
                self.accel = self.accel + self.wheel_force * self.radius / den
                # Friction (proportional to angular velocity. Increases with weight for now)
                self.accel = self.accel - self.velocity * self.friction

                self.clapper_accel = self.accel

                self.velocity = old_velocity + self.accel * phy.dt
                self.bell_angle = old_angle + self.velocity * phy.dt

                self.clapper_velocity = self.velocity
                # Update clapper angle
                self.clapper_angle = self.clapper_angle + self.clapper_velocity * phy.dt

        #Check if stay is broken
        if self.stay_hit > 0:
            self.stay_angle = 1e6
        # Check if bell has struck
        if self.clapper_angle - self.bell_angle < -self.clapper_limit:
            if self.ding_reset:
                self.volume_ref = 0.2 * abs(self.clapper_velocity - self.velocity)
            avg_velocity = (1 / (self.m_1 + self.m_2)) * (self.m_1 * self.velocity + self.m_2 * self.clapper_velocity)
            self.clapper_velocity = avg_velocity
            self.velocity = avg_velocity
            self.clapper_angle = -self.clapper_limit + self.bell_angle
            self.onedge = True

        elif self.clapper_angle - self.bell_angle > self.clapper_limit:
            if self.ding_reset:
                self.volume_ref = 0.2 * abs(self.clapper_velocity - self.velocity)
            avg_velocity = (1 / (self.m_1 + self.m_2)) * (self.m_1 * self.velocity + self.m_2 * self.clapper_velocity)
            self.clapper_velocity = avg_velocity
            self.velocity = avg_velocity
            self.clapper_angle = self.clapper_limit + self.bell_angle
            self.onedge = True
        else:
            self.onedge = False

        if self.onedge and self.ding_reset:
            #This is the strike time
            if phy.time > 0.1:

                if self.clapper_angle < -np.pi/4:
                    #print('Hand', phy.time, self.handstroke_target)
                    self.all_handstrokes.append(phy.time)
                    self.handstroke_accuracy.append(self.handstroke_target)
                    #UPDATE RHYTHM ROUTINES
                    if len(self.handstroke_accuracy) == 1 or self.update_rhythm:  #First handstroke -- establish rhythm
                        self.next_handstroke, self.next_backstroke = self.establish_rhythm(phy.time)
                        self.update_rhythm = False
                    else:
                        self.next_handstroke, self.next_backstroke = self.establish_rhythm(self.next_handstroke)
                    #print('Targets', self.next_backstroke, self.next_handstroke)
                elif self.clapper_angle > np.pi/4:
                    #print('Back', phy.time, self.backstroke_target)
                    self.all_backstrokes.append(phy.time)
                    self.backstroke_accuracy.append(self.backstroke_target)


            if phy.do_volume:
                if self.sound.get_volume() < self.volume_ref:
                    self.sound.set_volume(self.volume_ref)
                else:
                    self.sound.set_volume(self.volume_ref + self.sound.get_volume())
            self.ding = True
            self.ding_reset = False
            self.ding_time = phy.game_time
        else:
            self.ding = False

        if self.onedge and not self.ding_reset:
            if phy.do_volume:
                self.sound.set_volume(np.exp(-5e1 * phy.dt) * self.volume_ref)
        if abs(self.clapper_angle - self.bell_angle) < self.clapper_limit - 0.1:
            self.ding_reset = True

        self.rlength, self.effect_force = self.ropelength()
        self.rlengths.append(self.rlength)
        self.effect_forces.append(self.effect_force)

        if len(self.rlengths) > 3:  # Maximum height of previous backstroke. To allow for adjustment of tail end length.
            if self.effect_force > 0.0 and self.rlengths[-1] < self.rlengths[-2] and self.rlengths[-2] > self.rlengths[-3]:
                self.max_length = self.rlengths[-1]

        #Update stiking information
        self.last_backstroke = phy.time - self.all_backstrokes[-1]
        self.last_handstroke = phy.time - self.all_handstrokes[-1]
        phy.time = phy.time + phy.dt
        self.times.append(phy.time)
        self.bell_angles.append(self.bell_angle)
        self.velocities.append(self.velocity)
        self.forces.append(self.pull)

    def time_targets(self, phy):
        #Target ringing times, which can depend on the previous strike times.
        #Should not be negative for the inputs but can be here
        if False:  #Target is based on previous same stroke
            self.backstroke_target = self.all_backstrokes[-1] + self.target_period - phy.time
            self.handstroke_target = self.all_handstrokes[-1] + self.target_period - phy.time
        elif True:
            self.backstroke_target = self.all_handstrokes[-1] + self.nbells/(self.nbells*2 + 1)*self.target_period - phy.time
            self.handstroke_target = self.all_backstrokes[-1] + (self.nbells + 1)/(self.nbells*2 + 1)*self.target_period - phy.time
        else:  #Receive info from the rhythm function
            self.backstroke_target = self.next_backstroke - phy.time
            self.handstroke_target = self.next_handstroke - phy.time

    def ropelength(self):
        # Outputs the length of the rope above the garter hole, relative to the minimum.
        # Also outputs the maximum force available with direction.
        hole_angle = self.bell_angle - np.pi + self.garter_hole
        if hole_angle > 0.0:
            # Fully Handstroke
            length = self.radius * hole_angle + self.radius
            effect_force = -1.0
        elif hole_angle <= -np.pi / 2:
            # Fully backstroke
            length = self.radius * (-np.pi / 2 - hole_angle) + self.radius
            effect_force = 1.0
        else:
            # Somewhere in between
            xpos = self.radius + self.radius * np.sin(hole_angle)
            ypos = self.radius - self.radius * np.cos(hole_angle)
            vec1 = np.array([xpos, ypos])
            vec2 = np.array([np.cos(hole_angle), np.sin(hole_angle)])
            vec1 = vec1 / np.linalg.norm(vec1)
            vec2 = vec2 / np.linalg.norm(vec2)
            length = np.sqrt(xpos**2 + ypos**2)
            effect_force = -np.dot(vec1, vec2)

        return length, effect_force

    def get_scaled_state(self):
        """Get full system state, scaled into [0,1]."""
        """Angle then velocity (obviously veclocity can be large)"""
        bt = max(self.backstroke_target/10.0,0)   #Time until desired stroke
        ht = max(self.handstroke_target/10.0,0)
        pb = self.last_backstroke/10.0     #Time since last stroke
        ph = self.last_handstroke/10.0

        if self.bell_angle > np.pi:
            up_handstroke = (self.bell_angle-np.pi)/self.stay_angle
        else:
            up_handstroke = 0.0

        if self.bell_angle < -np.pi:
            up_backstroke = (-self.bell_angle-np.pi)/self.stay_angle
        else:
            up_backstroke = 0.0

        return [np.sin(self.bell_angle), np.cos(self.bell_angle), self.velocity*np.sign(self.bell_angle)/10.0, np.abs(self.possible_force), up_handstroke, up_backstroke]

        #return [self.bell_angle / (np.pi + self.stay_angle), self.velocity / (10.0), bt, ht, self.m_1/1000, pb, ph]

    def establish_rhythm(self, reference_time):
        """Estalishes the desired times for each stroke"""
        """Outputs the TIMES that these should happen as a series of arrays"""
        """Need to be able to readjust while ringing I suppose (but not for training)"""
        """Reference_time is the time of the previous first handstroke in the 'change'"""
        belltimes = np.linspace(reference_time, reference_time + self.target_period, self.nbells*2+2)
        #For steady rounds at the minute
        next_handstroke = belltimes[-1]
        next_backstroke = belltimes[self.nbells]

        return next_handstroke, next_backstroke

    def fitness_fn(self, phy, print_accuracy = False, verbose=False):
        """
        Fitness function not in increments. Look at the entire bit of ringing?
        """

        if self.current_mode == 'down':   #Ringing down overall performance -- only caring about end result once the rest has been optimised and the bell is ringing down reasonably well
            alpha = 2
            max_angle = 1e-1
            final_angle = (max(0., (max_angle - np.abs(self.bell_angle))/max_angle)**alpha)
            max_velocity = 1e-1
            final_velocity = (max(0., (max_velocity - np.abs(self.velocity))/max_velocity)**alpha)
            max_clapper = 0.6

            final_clapper_angle = (max(0., (max_clapper - np.abs(self.clapper_angle) - np.abs(self.clapper_velocity))/max_clapper)**alpha)

            max_force = 0.1
            final_force =( max(0., (max_force - np.abs(self.pull))/max_force)**alpha)
            if print_accuracy:
                print('Final stats', self.bell_angle, self.velocity, self.clapper_angle + self.clapper_velocity, self.pull)
                print('Final stats applied', final_angle, final_velocity, final_clapper_angle, final_force)
            return 0.4*final_angle + 0.4*final_velocity + 0.0*final_clapper_angle + 0.2*final_force

        elif self.current_mode == 'up':   #Ringing up overall performance -- time to up plus the force at the end?
            """
            As a first test, look at the maximum angle attained? Then minimise that, obviously
            """

            if False:
                #This seems to work. Need proportions to be well-defined. Let's also try to make 1.0 the maximum no matter what.
                cut = int(len(self.bell_angles)/4)

                max_angle = np.max(self.bell_angles[-cut:])
                max_angle = min(np.pi, max_angle)/np.pi

                angle_proportions = np.abs(self.bell_angles)/(np.pi + self.stay_angle)

                mean_angle = 1.0 - np.mean(angle_proportions**2)

                up_time = 0.0

                alpha = 2
                #Now look at time spent in the region of the balance - this is a decent metric
                #Time to set the bell
                for i in range(len(self.bell_angles)):
                    if self.bell_angles[i] > np.pi-0.05:
                        up_time += 1

                up_time = up_time/len(self.bell_angles)

                a0 = (1.0 - max_angle)**alpha   #The highest angle attained in the last quarter of the run
                a1 = np.mean(self.forces)**alpha  #The overall force proportion
                a2 = (1.0 - up_time)**alpha  #The first time it is up.

                #Change the proportions as appropriate based on the current skill?
                # if a0 < 0.2: #Bell is basically up. Don't care about the force's local minimum any more.
                #     a1 = 0.1*a1

                props = [1.0,0.15,0.0]
                props = props/np.sum(props)

                if verbose:
                    print('__________________')
                    print('Upness:', a0)
                    print('Forceness:', a1)
                    print('Timeliness:', a2)

            #New approach based on incremental improvements

            #Things to penalise:
            # 0: Angle. Is up-at-handstroke attained at any point. If not, reward the increase in swing magnitude
            # 1: backstroke penalty. Proportion of the time spent at backstroke. I think this is good enough to be stable now, but need to add on the extra time.
            # 2: handstroke penalty. Same deal
            # 3: stay penalty. Penalty for whacking the stay. Perhaps need to increase this a bit

            props = [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]
            #props = [1.0, 0.0, 0.0,0.0,0.0,0.0]

            max_time_cutoff = 20.0

            peaks, _ = find_peaks(np.abs(self.bell_angles))
            max_angles = np.array(np.abs(self.bell_angles))[peaks]

            if np.max(np.abs(self.bell_angles)) > 0.99*np.pi:  #Has got sufficiently up. Don't care about oscillations
                angle_penalty = 0.0
            elif len(peaks) < 2:
                angle_penalty = 1.0
            else:
                angle_penalty =  ((np.pi - max_angles[-1]) / (np.pi - 0.0))**2#max(0.1, max_angles[0]))

            if self.bell_angles[0] < -np.pi and self.bell_angles[-1] < -np.pi:
                angle_penalty = 1.0

            nsteps_overall = int(max_time_cutoff/phy.dt)
            tsteps_remaining = nsteps_overall - len(self.bell_angles)  #Amount of time left in the simulation
            at_backstroke = [1.0 if self.bell_angles[-1] < -np.pi else 0.0][0]

            backstroke_time = (np.sum(np.array(self.bell_angles) < -np.pi) + tsteps_remaining*at_backstroke)/nsteps_overall
            backstroke_penalty = backstroke_time**2 #/(1.0 - backstroke_time)

            at_handstroke = [1.0 if self.bell_angles[-1] > np.pi else 0.0][0]

            handstroke_time = (np.sum(np.array(self.bell_angles) > np.pi) + tsteps_remaining*at_handstroke)/nsteps_overall
            handstroke_penalty = (1.0 - handstroke_time)**2   #Encourage lingering at handstroke

            min_velocity = 0.25
            if self.stay_touch == 0:  #Never touches the stay, so assume it's fine
                stay_penalty = 1.0
            else:
                if self.stay_touch_velocity > min_velocity:
                    stay_hit_fraction = 0.125*(self.stay_touch_velocity - min_velocity)/self.stay_break_limit
                    stay_penalty = min(1.0, stay_hit_fraction**2)
                else:
                    stay_penalty = 0.0

            #Also would like to intoduce a penalty for force while set on the wrong stroke. Hopefully things will coincide and multiply easily.
            #Proportion of time over the balance at which the force is great
            set_hand = np.where(np.array(self.bell_angles) > np.pi + 0.1)[0]
            hand_forces = np.array(self.forces)[set_hand]
            if len(hand_forces) > 0:
                handforce_penalty = np.mean(np.abs(hand_forces)**2)
            else:
                handforce_penalty = 1.0 #Do want it to be at handstroke at some point

            set_back = np.where(np.array(self.bell_angles) < -np.pi - 0.1)[0]
            back_forces = np.array(self.forces)[set_back]
            if len(back_forces) > 0:
                backforce_penalty = 1.0 - np.mean(np.abs(back_forces)**2)
            else:
                backforce_penalty = 0.0 #Do want it to be at handstroke at some point

            if verbose:
                print('Penalties', angle_penalty,  stay_penalty, handstroke_penalty, backstroke_penalty, handforce_penalty, backforce_penalty)
            #return props[0]*a0 + props[1]*a1 + props[2]*a2
            return (props[0]*angle_penalty + props[1]*backstroke_penalty + props[2]*handstroke_penalty + props[3]*stay_penalty + props[4]*handforce_penalty + props[5]*backforce_penalty)/np.sum(props)

    def fitness_increment(self, phy):
        """Fitness function at a given time rather than evaulating after the fact"""
        """Must multiply by dt/tmax or equivalent"""
        mult = 60.0 * phy.FPS
        if self.current_mode == 'down':  # RINGING DOWN

            force_fraction = 0.1 #How much to care about the force applied at each stroke
            alpha = 2  #Distance factor

            downness = self.bell_angle**alpha
            forceness = self.pull**alpha
            fitness_increment = downness*(1.0 + force_fraction*forceness)

            return fitness_increment/mult

        elif self.current_mode == 'up':   #RINGING UP

            # if self.stay_hit > 0:  #Heavily penalise breaking a stay
            #     return 1e12

            if self.bell_angle > np.pi and self.stay_hit == 0:
                up_handstroke = True
            else:
                up_handstroke = False
            if self.bell_angle < -np.pi and self.stay_hit == 0:
                up_backstroke = True
            else:
                up_backstroke = False
            force_fraction = 0.0 #How much to care about the force applied at each stroke. Set to zero for now.
            alpha = 2  #Distance factor
            forceness = self.pull**alpha

            if self.bell_angle < -np.pi:   #Bell is set at the wrong stroke. Penalise accordingly and try not to sit there.
                upness = 10.0 - self.bell_angle - np.pi
            elif self.bell_angle < 0.0:
                upness = (np.pi - np.abs(self.bell_angle)/np.pi)**alpha
            elif self.bell_angle > np.pi: #Bell is correctly up
                upness = 0.0
            else:  #Bell is on the right side, but not up
                upness = (np.pi - np.abs(self.bell_angle)/np.pi)**alpha

            fitness_increment = upness*(1.0 + force_fraction*forceness)

            return fitness_increment/mult
        else:
            return 0.
