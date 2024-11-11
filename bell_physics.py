# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 17:29:08 2024

@author: eleph
"""
import numpy as np
import time
import pygame, sys
from pygame.locals import *


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
        self.max_length = 0.0  # max backstroke length

        self.rlength, self.effect_force = self.ropelength()
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
        self.stay_hit = False
        self.stay_break_limit = 0.5#1.5

        self.bell_angles = []
        self.forces = []
        self.times = [0.0]
        self.stay_hit = 0

    def timestep(self, phy):
        # Do the timestep here, using only bell.force, which comes either from an input or the machine
        # Update the physics here
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
            if abs(self.velocity) < 0.01 and self.wheel_force == 0.0:
                self.velocity = 0.5 * self.velocity

            self.prev_angle = self.bell_angle
            self.bell_angle = self.bell_angle + self.velocity * phy.dt

            # check if stay has been hit, and bounce if so
            if self.bell_angle > np.pi + self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = 2 * np.pi + 2 * self.stay_angle - self.bell_angle
                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity
            if self.bell_angle < -np.pi - self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = -2 * np.pi - 2 * self.stay_angle - self.bell_angle

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
            if abs(self.velocity) < 0.01 and self.wheel_force == 0.0:
                self.velocity = 0.5 * self.velocity

            self.prev_angle = self.bell_angle
            self.bell_angle = self.bell_angle + self.velocity * phy.dt

            # check if stay has been hit, and bounce if so
            if self.bell_angle > np.pi + self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = 2 * np.pi + 2 * self.stay_angle - self.bell_angle
                if abs(self.velocity) > self.stay_break_limit:
                    self.stay_hit = self.stay_hit + 1
                    self.velocity = -0.5 * self.velocity

            if self.bell_angle < -np.pi - self.stay_angle:
                self.velocity = -0.7 * self.velocity
                self.bell_angle = -2 * np.pi - 2 * self.stay_angle - self.bell_angle
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

        phy.time = phy.time + phy.dt
        self.times.append(phy.time)
        self.bell_angles.append(self.bell_angle)
        self.forces.append(self.wheel_force)

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
        return [self.bell_angle / (np.pi + self.stay_angle), self.velocity / (10.0)]

    def fitness_fn(self):
        """Define fitness function, only of variables of 'self'"""
        if False:  # RINGING UP WITHOUT STAY HITS
            angle_aim = np.pi
            absangles = np.array(np.abs(self.bell_angles))
            max_travel = np.pi + self.stay_angle
            return np.sum((max_travel - np.abs(angle_aim - absangles)) ** 2 / max_travel**2) / len(np.array(self.bell_angles))
        if False:  # RINGING UP WITH STAY HITS
            angle_aim = np.pi
            absangles = np.array(np.abs(self.bell_angles))
            max_travel = np.pi + self.stay_angle

            return np.sum(np.array(self.bell_angles) ** 2 / np.pi**2) / len(np.array(self.bell_angles)) / (self.stay_hit + 1)

        if True:  # RINGING DOWN
            angle_aim = 0.0
            absangles = np.abs((np.pi + self.stay_angle) - np.array(np.abs(self.bell_angles)))

            max_angle = np.max(np.abs(self.bell_angles))
            max_travel = np.pi + self.stay_angle

            #Penalise for amount of time pulling?
            forcetime = max(np.sum([np.array(self.forces) > 10])/len(self.forces), 0.5)
            alpha = 4
            return np.sum(absangles**alpha/max_travel**alpha) / (len(np.array(self.bell_angles)) * (self.stay_hit + 1))


        #Score in the moment -- in a random 10 second window is the bell now further down than at the start, and by how much?
        if False:
            nangles = len(self.bell_angles)
            maxheight_begin = np.max(np.abs(self.bell_angles[:nangles//2]))
            maxheight_end = np.max(np.abs(self.bell_angles[nangles//2:]))
            if maxheight_begin < 1e-6 or maxheight_end  < 1e-6:
                return 1e6
            if self.stay_hit < 1:
                return maxheight_begin/maxheight_end
            else:
                return 0.0












