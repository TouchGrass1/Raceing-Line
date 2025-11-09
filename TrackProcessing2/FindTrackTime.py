import generateSpline
import pygame as pg
import time
from enum import Enum
from pygame.locals import *
from Physics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize



def findMaxVelocity(radius, mass):
   
    downforce = 2000 #N
    mu = 1.5
    maxLateralForceEquation = PhysicsFormulas.maxLateralForceEquation(mu, mass, downforce)
    maxVel = np.sqrt((maxLateralForceEquation*radius) / mass)
    return maxVel

def findVelocities(maxVelArr, mass):
    n = len(maxVelArr)
    vel = np.zeros(n)
    
    # Forward pass – acceleration limit
    vel[0] = maxVelArr[0]
    for i in range(1, n):
        # Accelerate but cap by max velocity
        a = min(PhysicsFormulas.accelerationPowerEquation(vel[-1], mass), PhysicsConsts['ACCEL_MAX'].value)
        vel[i] = min(maxVelArr[i], vel[i-1] + a)

    # Backward pass – deceleration limit
    for i in range(n-2, -1, -1):
        # Ensure deceleration limit isn’t exceeded
        if vel[i] > vel[i+1] - PhysicsConsts['ACCEL_MIN'].value:
            vel[i] = vel[i+1] - PhysicsConsts['ACCEL_MIN'].value
        # Also respect the original max velocity
        vel[i] = min(vel[i], maxVelArr[i])

    return vel

def calculateTrackTime(vel, rand_bsp, pixels_per_meter):
    x, y = rand_bsp[:, 0], rand_bsp[:, 1]

    # Compute distances between consecutive points
    distances = np.hypot(x, y)
    t = np.concatenate(([0], np.cumsum(distances/vel)))
    return t[-1]
        

def plotVelocity(vel, maxVel, radius):

    x = np.arange(len(vel))
    plt.plot(x, vel)
    plt.plot(x, maxVel)
    plt.plot(x, radius)
    plt.show()



def plot_velocity_colored_line(spline_points, velocities, cmap='turbo'):
    """
    Plot racing line with velocities color-coded along its length.
    
    spline_points: Nx2 array of (x, y) coordinates.
    velocities: array of velocity values (same length as spline_points).
    """
    # Make sure both are numpy arrays
    spline_points = np.asarray(spline_points)
    velocities = np.asarray(velocities)

    # Create segments between consecutive points
    points = spline_points.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Normalize velocities to colormap range
    norm = Normalize(vmin=np.min(velocities), vmax=np.max(velocities))

    # Create a line collection
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(velocities)
    lc.set_linewidth(3)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal', 'box')

    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax)
    cbar.set_label('Velocity (m/s)', fontsize=12)

    ax.set_title("Velocity-Colored Racing Line", fontsize=14)
    plt.show()


def main(rand_bsp, radius, mass, pixels_per_meter):



    maxVelArr = findMaxVelocity(radius, mass)
    vel = findVelocities(maxVelArr, mass)
    t  = calculateTrackTime(vel, rand_bsp, pixels_per_meter)
    return vel, t


#vel, rand_bsp, t = main()
#print('time: ', t)

#plotVelocity(vel, maxVelArr, mean_radius)
#plot_velocity_colored_line(rand_bsp, vel)



#generateSpline.plot_spline(rand_bsp, random_pts, None)
#generateSpline.plot_bspline(rand_bsp, random_pts, mesh, props["curvature"])
#generateSpline.plot_everything(mesh, center_line, center_line_ctrpts, rand_bsp, random_pts)

run = True
#while run:


