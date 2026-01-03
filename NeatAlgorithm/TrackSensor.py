import numpy as np
from math import hypot, atan2, cos, sin


def nearest_index_on_line(point, line_pts):
    # simple nearest-point (could be sped up with KDTree)
    dists = np.sum((line_pts - np.array(point))**2, axis=1)
    idx = int(np.argmin(dists))
    return idx

def track_progress(position, center_line):
    idx = nearest_index_on_line(position, center_line)
    return idx / len(center_line)

def ray_distance(position, angle_rad, max_dist, centerline=None, inner=None, outer=None, step=1.0):
    #cast a circle around the point with increasing radii
    x0, y0 = position
    cos_a = cos(angle_rad)
    sin_a = sin(angle_rad)
    nsteps = int(max_dist / step)
    for i in range(1, nsteps + 1):
        x = x0 + cos_a * i * step
        y = y0 + sin_a * i * step
        # check if (x,y) is outside track region between inner and outer
        if centerline is not None and inner is not None and outer is not None:
            idx = nearest_index_on_line((x,y), centerline)
            local_half_width = np.linalg.norm((outer[idx] - inner[idx])) / 2.0
            dist_to_center = np.linalg.norm(np.array((x,y)) - centerline[idx])
            if dist_to_center > local_half_width + 0.5:  # small tolerance
                return i * step
    return max_dist

def make_ray_sensors(position, heading_rad, center_line, inner, outer, n_rays=5, fov=np.pi/2, max_dist=200.0):
    # fov centered on heading; returns distances of each ray
    angles = []
    dists = []
    for i in range(n_rays):
        angles.append(heading_rad + (i/(n_rays-1) - 0.5) * fov )
    for a in angles:
        dists.append(ray_distance(position, a, max_dist, center_line, inner, outer, step=2.0))
    return dists, angles