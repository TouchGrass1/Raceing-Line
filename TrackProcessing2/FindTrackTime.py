#import generateSpline

from Physics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import generateSpline


def findMaxVelocity(radius, mass, density):
   
    downforce = 2000 #N
    mu = 1.5
    maxVerticalThrust, maxVerticalVel = PhysicsFormulas.maxThrustEquation(density)
    maxLateralForceEquation = PhysicsFormulas.maxLateralForceEquation(mu, mass, downforce)
    maxLateralForceEquation = PhysicsFormulas.maxLateralAccelerationEquation(maxLateralForceEquation, mass)
    maxVel = np.sqrt((maxLateralForceEquation*radius) / mass)
    np.clip(maxVel, 0, maxVerticalVel)
    return maxVel

def findVelocities(maxVelArr, rand_bsp, pixels_per_meter, mass, density, noLap, tyreType):
    n = len(maxVelArr)
    vel = np.zeros(n)
    rand_bsp = np.vstack([rand_bsp, rand_bsp[0]])
    x, y = rand_bsp[:, 0], rand_bsp[:, 1]
    dx = np.hypot(np.diff(x), np.diff(y))
    dx = dx/pixels_per_meter

    # Forward pass acceleration limit
    vel[0] = 1
    for i in range(1, n):
        u = vel[i-1]
        downForce = updateVar.updateDownforce(u)
        thrust = updateVar.updateThrust(u)
        drag = updateVar.updateDrag(u, density)
        finalForce = updateVar.updateResultantForce(thrust, drag)
        tyreCoeff = updateVar.updateTyreWear(tyreType, noLap)
        staticFriction = updateVar.updateStaticForceFriction(tyreCoeff, mass, downForce)

        # Accelerate but cap by max velocity
        a = min(staticFriction/mass, PhysicsConsts['ACCEL_MAX'].value, finalForce/mass)

        coeff = [0.5*a, u, -dx[i]]
        dt = max(np.roots(coeff))
        vel[i] = min(maxVelArr[i], u + a*dt)

    # Backward pass deceleration limit
    for i in range(n-2, -1, -1):
        # Ensure deceleration limit isn’t exceeded
        v = vel[i+1]
        u = vel[i]
        dist = dx[i]

        max_decel = PhysicsConsts['ACCEL_MIN'].value 
        
        u = np.sqrt(v**2 - 2 * max_decel * dist)
        
        vel[i] = min(vel[i], u)


    return vel

def calculateTrackTime(vel, rand_bsp, pixels_per_meter):
    x, y = rand_bsp[:, 0], rand_bsp[:, 1]

    distances = np.hypot(np.diff(x), np.diff(y)) #pythag
    distances = distances/pixels_per_meter
    t = np.concatenate(([0], np.cumsum(distances/vel[:len(vel)-1])))
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



def main(rand_bsp, radius, pixels_per_meter):
    
    mass = 900 #KG
    maxNoLap = 70 #maximum number of laps
    #CONSTANTS - set by user
    temp = 20 #deg
    height = 300 #m above sea level
    pressure = updateVar.updatePressure(height, temp)
    density = updateVar.updateDensity(pressure, temp)
    tyreType = 'SOFT'

    #EVERY  LAP STUFF
    noLap = 1 #first lap
    mass = updateVar.updateMass(noLap, maxNoLap)
    #update tyre wear...


    maxVelArr = findMaxVelocity(radius, mass, density)
    vel = findVelocities(maxVelArr, rand_bsp, pixels_per_meter, mass, density, noLap, tyreType)
    t  = calculateTrackTime(vel, rand_bsp, pixels_per_meter)
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


def init_track():
    global track_name
    track_list = ["monza", "silverstone", "qatar", "90degturn"]
    track_name = track_list[1]
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
    #print(f"time taken: {int(t[-1]//60)}: {t[-1]%60:.3f}")
    

    #plot_velocity_coloured_line_v2(racing_line, vel, mesh)
    #plotVelocity(vel, a)



run()
#generateSpline.plot_spline(rand_bsp, random_pts, None)
#generateSpline.plot_bspline(rand_bsp, random_pts, mesh, props["curvature"])
#generateSpline.plot_everything(mesh, center_line, center_line_ctrpts, rand_bsp, random_pts)