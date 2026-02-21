import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import TrackProcessing2.generateSpline as generateSpline
from TrackProcessing2.config import variable_options, real_properties
from TrackProcessing2.Physics import *
# from Physics import *
# import generateSpline as generateSpline
# from config import variable_options, real_properties


def findMaxVelocity(radius, mass, density, mu):
   
    downforce = 2000 #N
    maxVerticalThrust, maxVerticalVel = PhysicsFormulas.maxThrustEquation(density)
    maxLateralForceEquation = PhysicsFormulas.maxLateralForceEquation(mu, mass, downforce)
    maxLateralForceEquation = PhysicsFormulas.maxLateralAccelerationEquation(maxLateralForceEquation, mass)
    maxVel = np.sqrt((maxLateralForceEquation*radius) / mass)
    np.clip(maxVel, 0, maxVerticalVel)
    return maxVel

def findVelocities(maxVelArr, dists, mass, density, noLap, tyreType):
    n = len(dists)

    vel = np.zeros(n)

    # Forward pass acceleration limit
    if maxVelArr[0] < PhysicsConsts['VELOCITY_MAX'].value: vel[0] = maxVelArr[0]
    else: vel[0] = PhysicsConsts['VELOCITY_MAX'].value
        
    for i in range(1, n):
        u = vel[i-1]
        downForce = updateVar.updateDownforce(u)
        thrust = updateVar.updateThrust(u)
        drag = updateVar.updateDrag(u, density)
        finalForce = updateVar.updateResultantForce(thrust, drag)
        tyreCoeff = updateVar.updateTyreWear(tyreType, noLap)
        staticFriction = updateVar.updateStaticForceFriction(tyreCoeff, mass, downForce)

        # Accelerate but cap by max velocity
        a_max = min(staticFriction/mass, PhysicsConsts['ACCEL_MAX'].value, finalForce/mass)
        vel[i] = min(maxVelArr[i], np.sqrt(u**2 + 2 * a_max * dists[i-1]))

    # Backward pass deceleration limit
    for i in range(n-1, -1, -1):
        # Ensure deceleration limit isn’t exceeded
        next_idx = (i + 1) % n
        v_next = vel[next_idx]
        dist = dists[i]

        max_decel = PhysicsConsts['ACCEL_MIN'].value 
        # u^2 = v^2 - 2as (where a is negative)
        possible_u = np.sqrt(v_next**2 - 2 * max_decel * dist)
        vel[i] = min(vel[i], possible_u)


    return vel

def calculateTrackTime(vel, dists):

    avg_vels = (vel + np.roll(vel, -1)) / 2 #using roll as it faster than looping and ensures circular
    segment_times = dists / avg_vels
    
    t = np.concatenate(([0], np.cumsum(segment_times))) #cumulative times

    return t

def caculateAccelerations(vel, t):     
    a_arr = np.zeros(len(vel))
    for i in range(1, len(vel)):
        dt = t[i] - t[i-1]
        if dt == 0:
            a_arr[i] = 0
        else:
            
            a_arr[i] = (vel[i] - vel[i-1]) / dt
    return a_arr

def plotVelocity(vel, a_arr):

    x = np.arange(len(vel))
    plt.plot(x, vel)
    plt.plot(x, a_arr)
    #plt.plot(x, radius)
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

def plot_velocity_coloured_line_v2(points, velocities, mesh):
    x = points[:,0]
    y = points[:,1]

    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]

    # Each line seg = [(x0,y0),(x1,y1)]
    segments = np.stack([np.column_stack([x[:-1], y[:-1]]),
                         np.column_stack([x[1:],  y[1:]])], axis=1)

    fig, ax = plt.subplots(figsize=(7,7))

    lc = LineCollection(segments, cmap='turbo', linewidth=4)
    lc.set_array(velocities[:-1])        # colour by speeds
    line = ax.add_collection(lc)

    cbar = fig.colorbar(line, ax=ax)
    cbar.set_label("Velocity (m/s)")

    ax.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    ax.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')

    ax.set_aspect('equal')
    ax.set_title("Racing Line Colour-Graded by Velocity")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.autoscale()

    plt.show()



def main(rand_bsp, radius, pixels_per_meter, variables): 
    mass = variables['mass'] #KG
    temp = variables['temp'] #deg
    height = variables['elevation'] #m above sea level
    noLap = variables['lapNo']
    tyreType = variables['tyre']


    maxNoLap = 70 #maximum number of laps
    
    pressure = updateVar.updatePressure(height, temp)
    density = updateVar.updateDensity(pressure, temp)

    tyre_mu = updateVar.updateTyreWear(tyreType, noLap)
    if variables['weather'] == 'wet':
        tyre_mu = tyre_mu / 2

    x_loop = np.append(rand_bsp[:, 0], rand_bsp[0, 0])
    y_loop = np.append(rand_bsp[:, 1], rand_bsp[0, 1])
    dists = np.hypot(np.diff(x_loop), np.diff(y_loop)) / pixels_per_meter
    #EVERY  LAP STUFF
    
    mass = updateVar.updateMass(noLap, maxNoLap)


    maxVelArr = findMaxVelocity(radius, mass, density, tyre_mu)

    vel = findVelocities(maxVelArr, dists, mass, density, noLap, tyreType)
    t  = calculateTrackTime(vel, dists)
    print(f"lap time: {int(t[-1]//60)}: {t[-1]%60:.3f}")
    return vel, t



def sample_90deg_racing_line(num_points=200, R=50, entry=2000, exit=20):
    pts = []

    # Entry straight (along +x direction)
    xs = np.linspace(0, entry, num_points//4)
    ys = np.zeros_like(xs)
    pts.extend(np.column_stack((xs, ys)))

    # 90-degree left arc
    thetas = np.linspace(0, np.pi/2, num_points//2)
    xc, yc = entry, R  # circle center
    xs = xc + R * np.sin(thetas)
    ys = yc - R * np.cos(thetas)
    pts.extend(np.column_stack((xs, ys)))

    # Exit straight (heading upward)
    xs = np.full(num_points//4, entry + R)
    ys = np.linspace(R, R + exit, num_points//4)
    pts.extend(np.column_stack((xs, ys)))

    return np.array(pts)

def radii_for_90deg(num_points=200, R=50):
    arr = np.zeros(num_points)
    arr[:num_points//4] = np.inf       # entry straight
    arr[num_points//4:3*num_points//4] = R
    arr[3*num_points//4:] = np.inf     # exit straight
    return arr

# racing_line = sample_90deg_racing_line()
# radius = radii_for_90deg()
# pixels_per_meter = 0.4
# vel, t = main(racing_line, radius, pixels_per_meter)



def init_track():
    global track_name
    track_list = variable_options['track']
    track_name = track_list[2]
    return generateSpline.main(track_name, real_properties)

def create_random_bsp(mesh, track_name, center_line_properties):
    random_pts = generateSpline.random_points(mesh, num_pts_across=50, rangepercent=0.020, sample_size=500)
    rand_bsp = generateSpline.catmull_rom_spline(random_pts)
    props = generateSpline.spline_properties(rand_bsp)

    pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']

    # compute radius (1/curvature), skip zero curvature to avoid div by zero
    radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]

    radius = np.array(radius) / pixels_per_meter
    return rand_bsp, radius

def run():
    center_line_ctrpts, center_line, center_line_properties, mesh = init_track()
    pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']
    racing_line, radius = create_random_bsp(mesh, track_name, center_line_properties)
    vel, t = main(racing_line, radius, pixels_per_meter)
    #a = caculateAccelerations(vel, t)
    print(f"time taken: {int(t[-1]//60)}: {t[-1]%60:.3f}")
    

    plot_velocity_coloured_line_v2(racing_line, vel, mesh)
    #plotVelocity(vel, a)



#run()
#generateSpline.plot_spline(rand_bsp, random_pts, None)
#generateSpline.plot_bspline(rand_bsp, random_pts, mesh, props["curvature"])
#generateSpline.plot_everything(mesh, center_line, center_line_ctrpts, rand_bsp, random_pts)