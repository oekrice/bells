# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 10:03:56 2024

@author: eleph
"""


import numpy as np

from bell_physics import init_bell, init_physics

import matplotlib.pyplot as plt
import random

class run_bell(object):
    
    def __init__(self):
        self.phy = init_physics()
        self.bell = init_bell(self.phy, 0.0)
        
        self.wheel_force = 600.   #max force on the rope (in Newtons)
        self.count = 0
        self.max_time = 120.
        self.phy.do_volume = False
        
    def step(self, force):
        #Calculate the force exactly as in the playable version
        if self.bell.effect_force < 0.0:   #Can pull the entire handstroke
            self.bell.wheel_force = self.bell.effect_force*self.wheel_force
        else:           #Can only pull some of the backstroke
            if self.bell.rlength > self.bell.max_length - self.bell.backstroke_pull:
                self.bell.wheel_force = self.bell.effect_force*self.wheel_force
            else:
                self.bell.wheel_force = 0.0
        #Does a single timestep on the stuff in the bell class
        self.bell.timestep(self.phy)
        self.phy.count = self.phy.count + 1
                    
        self.count = self.count + 1
        
    def get_scaled_state(self):
        """Get full system state, scaled into [0,1]."""
        """Angle then velocity (obviously veclotiy can be large)"""
        return [self.bell.bell_angle/(np.pi + self.bell.stay_angle), self.bell.velocity/(1.0)]

def continuous_actuator_force(action):
    return action[0]

def discrete_actuator_force(action):
    if action[0] > 0.5:
        return 1.
    else:
        return 0.

def probably_actuator_force(action):
    if action[0] > random.random():
        return 1.
    else:
        return 0.



