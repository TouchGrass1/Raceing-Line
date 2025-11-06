import generateSpline
import pygame as pg
import time
from enum import Enum
from pygame.locals import *
from Physics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

#CONSTANTS
class colour_palette(Enum):
    WHITE = (242, 241, 242)
    RED = (246, 32, 57)
    BLACK = (17, 17, 17)
    BLUE = (41, 82, 148)
    BG_GREY = (2, 29, 33)
    ORANGE = (255, 128, 0)
    RED2 = (187, 57, 59)
    LINE_GREY = (42, 45, 49)
    SUBTLE_GREY = (37, 40, 44)

real_properties = {
'silverstone': {
    'real_track_length': 5891, #meters
    'real_track_width': 12 #meters
    },
'monza': {
    'real_track_length': 5793,
    'real_track_width': 12
    },
'qatar': {
    'real_track_length': 5419,
    'real_track_width': 12
    }
}

def findMaxVelocity(radius):
    mass = 700 #kg
    downforce = 2000 #N
    mu = 1.5
    maxLateralForceEquation = PhysicsFormulas.maxLateralForceEquation(mu, mass, downforce)
    maxVel = np.sqrt((maxLateralForceEquation*radius) / mass)
    return maxVel

def findVelocities(maxVelArr):
    n = len(maxVelArr)
    vel = np.zeros(n)
    
    # Forward pass – acceleration limit
    vel[0] = maxVelArr[0]
    for i in range(1, n):
        # Accelerate but cap by max velocity
        vel[i] = min(maxVelArr[i], vel[i-1] + PhysicsConsts['ACCEL_MAX'].value)

    # Backward pass – deceleration limit
    for i in range(n-2, -1, -1):
        # Ensure deceleration limit isn’t exceeded
        if vel[i] > vel[i+1] - PhysicsConsts['ACCEL_MIN'].value:
            vel[i] = vel[i+1] - PhysicsConsts['ACCEL_MIN'].value
        # Also respect the original max velocity
        vel[i] = min(vel[i], maxVelArr[i])

    return vel


        

def plotVelocity(vel, maxVel, radius):

    x = np.arange(len(vel))
    plt.plot(x, vel)
    plt.plot(x, maxVel)
    plt.plot(x, radius)
    plt.show()



def main():
    #SPLINE STUFF
    return 1

track_list = ["monza", "silverstone", "qatar"]
track_name = track_list[1]

center_line_ctrpts, center_line, center_line_properties, mesh = generateSpline.main(track_name, real_properties)

random_pts = generateSpline.random_points(mesh, num_pts_across=50, rangepercent=0.000)

rand_bsp, curvature = generateSpline.b_spline(random_pts, sample_size= 7000)
radius = 1/abs(curvature)
pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']
radius = radius / pixels_per_meter #conert to meters

maxVelArr = findMaxVelocity(radius)
vel = findVelocities(maxVelArr)



plotVelocity(vel, maxVelArr, radius)
generateSpline.plot_spline(rand_bsp, random_pts, None)

run = True
#while run:


