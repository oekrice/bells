# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:03:56 2024

@author: eleph
"""

import numpy as np
import pygame, sys
from pygame.locals import *
import time


class display_tools:
    # Display tools, to keep them out of the way of the main function

    def __init__(self, phy, bell):
        self.surface = pygame.display.set_mode((phy.pixels_x, phy.pixels_y), 0, 32)

    def define_colours(self):
        self.WHITE = (255, 255, 255)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.BLACK = (0, 0, 0)
        self.DARKBROWN = (142, 62, 0)
        self.LIGHTBROWN = (245, 211, 120)

    def import_images(self, phy, bell):
        self.wheelimg = pygame.image.load("wheel.png")
        self.wheelimg = pygame.transform.scale(self.wheelimg, (2 * bell.radius * phy.xscale, 2 * bell.radius * phy.yscale))

        self.clapperimg = pygame.image.load("clapper.png")
        self.clapperimg = pygame.transform.scale(self.clapperimg, (2 * bell.radius * phy.xscale, 2 * bell.radius * phy.yscale))

        self.justbellimg = pygame.image.load("justbell.png")
        self.justbellimg = pygame.transform.scale(self.justbellimg, (2 * bell.radius * phy.xscale, 2 * bell.radius * phy.yscale))

    def display_stroke(self, phy, bell):
        # Display 'handstroke' or 'backstroke'
        fontObj = pygame.font.Font(pygame.font.match_font("arial"), 16)
        if bell.effect_force < 0.0:
            textSurfaceObj = fontObj.render("Handstroke", True, self.BLACK, self.WHITE)
        elif bell.rlength > bell.max_length - bell.backstroke_pull:
            textSurfaceObj = fontObj.render("Backstroke", True, self.BLACK, self.WHITE)
        else:
            textSurfaceObj = fontObj.render("", True, self.BLACK, self.WHITE)
        textRectObj = textSurfaceObj.get_rect()
        textRectObj.center = (0.8 * phy.pixels_x, 0.8 * phy.pixels_y)

        self.surface.blit(textSurfaceObj, textRectObj)

    def display_state(self, phy, ring_up, ring_down):
        # Display 'handstroke' or 'backstroke'
        fontObj = pygame.font.Font(pygame.font.match_font("arial"), 16)
        if ring_up:
            textSurfaceObj = fontObj.render("Ringing up", True, self.BLACK, self.WHITE)
        else:
            textSurfaceObj = fontObj.render("Ring up", True, self.BLACK, self.WHITE)
        textRectObj = textSurfaceObj.get_rect()
        textRectObj.center = (0.2 * phy.pixels_x, 0.2 * phy.pixels_y)
        self.surface.blit(textSurfaceObj, textRectObj)

        if ring_down:
            textSurfaceObj = fontObj.render("Ringing down", True, self.BLACK, self.WHITE)
        else:
            textSurfaceObj = fontObj.render("Ring down", True, self.BLACK, self.WHITE)
        textRectObj = textSurfaceObj.get_rect()
        textRectObj.center = (0.8 * phy.pixels_x, 0.2 * phy.pixels_y)

        self.surface.blit(textSurfaceObj, textRectObj)

    def display_force(self, phy, bell, force):
        # Display 'handstroke' or 'backstroke'
        fontObj = pygame.font.Font(pygame.font.match_font("arial"), 16)
        if bell.stay_hit == 0:
            textSurfaceObj = fontObj.render("Force on wheel = %03d" % force, True, self.BLACK, self.WHITE)
            textRectObj = textSurfaceObj.get_rect()
            textRectObj.center = (0.5 * phy.pixels_x, 0.9 * phy.pixels_y)
            self.surface.blit(textSurfaceObj, textRectObj)
        else:
            textSurfaceObj = fontObj.render("Stay Broken. Click to reset.", True, self.BLACK, self.WHITE)
            textRectObj = textSurfaceObj.get_rect()
            textRectObj.center = (0.5 * phy.pixels_x, 0.9 * phy.pixels_y)
            self.surface.blit(textSurfaceObj, textRectObj)

    def draw_bell(self, phy, bell):
        # Roate the bell image and paste
        wheel_rot, (x_box, y_box) = phy.rotate(self.wheelimg, bell.bell_angle)
        self.surface.blit(wheel_rot, (phy.pix(x_box, y_box)))

        clapper_rot, (x_box, y_box) = phy.rotate(self.clapperimg, bell.clapper_angle)
        self.surface.blit(clapper_rot, (phy.pix(x_box, y_box)))

        clapper_rot, (x_box, y_box) = phy.rotate(self.justbellimg, bell.bell_angle)
        self.surface.blit(clapper_rot, (phy.pix(x_box, y_box)))

    def draw_rope(self, phy, bell):

        # DO THIS NICELY - x and y are the CORRECT way up. Always use these functions to display.
        ppix = lambda p: (phy.pixels_x / 2 + p[0] * phy.xscale, (phy.pixels_y / 2 - p[1] * phy.yscale))

        # Figure out rope coordintes. Needs to draw as a polygon, which is a bit of a faff.
        hole_angle = bell.bell_angle - np.pi + bell.garter_hole  # angle of garter hole from bottom centre
        rope_edge = 0.04  # distance the inside of the rope is from the wheel edge
        rope_width = rope_edge / 2
        box_width = 0.05
        # Coordinates are the locations of the CENTRE of the rope
        xbase = -(bell.radius + rope_width / 2 - rope_edge)
        ybase = -(bell.radius + rope_width / 2 - rope_edge)

        if hole_angle > 0.0:
            # Fully handstroke
            xwheel = np.sin(0.0) * (bell.radius + rope_width / 2 - rope_edge)
            ywheel = -np.cos(0.0) * (bell.radius + rope_width / 2 - rope_edge)
            rope_angle = 0.0

        elif hole_angle <= -np.pi / 2:
            # Fully Backstroke
            rope_angle = np.pi / 2
            xwheel = np.sin(-np.pi / 2) * (bell.radius + rope_width / 2 - rope_edge)
            ywheel = -np.cos(-np.pi / 2) * (bell.radius + rope_width / 2 - rope_edge)
            rope_angle = np.pi / 2

        else:
            # Somewhere in between. Have to do some horrid transformations because of coordinates.
            xwheel = np.sin(hole_angle) * (bell.radius + rope_width / 2 - rope_edge)
            ywheel = -np.cos(hole_angle) * (bell.radius + rope_width / 2 - rope_edge)
            rope_angle = np.arctan2(ywheel - ybase, xwheel - xbase)

        # Rectangle coordinates
        p0 = [xbase + rope_width * np.sin(rope_angle) / 2, ybase - rope_width * np.cos(rope_angle) / 2]
        p1 = [xbase - rope_width * np.sin(rope_angle) / 2, ybase + rope_width * np.cos(rope_angle) / 2]

        p3 = [xwheel + rope_width * np.sin(rope_angle) / 2, ywheel - rope_width * np.cos(rope_angle) / 2]
        p2 = [xwheel - rope_width * np.sin(rope_angle) / 2, ywheel + rope_width * np.cos(rope_angle) / 2]

        pygame.draw.polygon(self.surface, self.LIGHTBROWN, [ppix(p0), ppix(p1), ppix(p2), ppix(p3)])
        pygame.draw.polygon(self.surface, self.BLACK, [ppix(p0), ppix(p1), ppix(p2), ppix(p3)], width=1)

        # Base rope coordinates
        p0 = [xbase + rope_width * np.sin(np.pi / 2) / 2, ybase - rope_width * np.cos(np.pi / 2) / 2]
        p1 = [xbase - rope_width * np.sin(np.pi / 2) / 2, ybase + rope_width * np.cos(np.pi / 2) / 2]

        p2 = [xbase + rope_width * np.sin(np.pi / 2) / 2, ybase - rope_width * np.cos(np.pi / 2) / 2 - 10.0]
        p3 = [xbase - rope_width * np.sin(np.pi / 2) / 2, ybase + rope_width * np.cos(np.pi / 2) / 2 - 10.0]

        pygame.draw.polygon(self.surface, self.LIGHTBROWN, [ppix(p0), ppix(p1), ppix(p2), ppix(p3)])
        pygame.draw.polygon(self.surface, self.BLACK, [ppix(p0), ppix(p1), ppix(p2), ppix(p3)], width=1)

        p0 = [xbase - box_width / 2, ybase - box_width / 2]
        p1 = [xbase - box_width / 2, ybase + box_width / 2]
        p2 = [xbase + box_width / 2, ybase + box_width / 2]
        p3 = [xbase + box_width / 2, ybase - box_width / 2]

        pygame.draw.polygon(self.surface, self.DARKBROWN, [ppix(p0), ppix(p1), ppix(p2), ppix(p3)])
        pygame.draw.polygon(self.surface, self.BLACK, [ppix(p0), ppix(p1), ppix(p2), ppix(p3)], width=1)

        # pygame.draw.circle(self.surface, self.DARKBROWN, (xpix(xbase), ypix(ybase)),8,0)
        # pygame.draw.circle(self.surface, self.BLACK, (xpix(xbase), ypix(ybase)),8,2)

        return
