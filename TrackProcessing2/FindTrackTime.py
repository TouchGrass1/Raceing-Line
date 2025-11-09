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
    },
'90degturn': {
    'real_track_length': 1000,
    'real_track_width': 12
    }
}

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


def main():


    track_list = ["monza", "silverstone", "qatar", "90degturn"]
    track_name = track_list[1]

    center_line_ctrpts, center_line, center_line_properties, mesh = generateSpline.main(track_name, real_properties)

    random_pts = generateSpline.random_points(mesh, num_pts_across=50, rangepercent=0.020, sample_size=500)
    rand_bsp = generateSpline.catmull_rom_spline(random_pts)
    props = generateSpline.spline_properties(rand_bsp)

    pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']

    # compute radius (1/curvature), skip zero curvature to avoid div by zero
    radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]

    radius = np.array(radius) / pixels_per_meter




    mass = 700 #kg
    maxVelArr = findMaxVelocity(radius, mass)
    vel = findVelocities(maxVelArr, mass)
    return random_pts, vel, rand_bsp


#random_pts, vel, rand_bsp = main()
#plotVelocity(vel, maxVelArr, mean_radius)
#plot_velocity_colored_line(rand_bsp, vel)



#generateSpline.plot_spline(rand_bsp, random_pts, None)
#generateSpline.plot_bspline(rand_bsp, random_pts, mesh, props["curvature"])
#generateSpline.plot_everything(mesh, center_line, center_line_ctrpts, rand_bsp, random_pts)

run = True
#while run:


