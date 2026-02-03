import cv2
import numpy as np
import random
from math import ceil

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from skimage.morphology import skeletonize
from scipy import interpolate



from pathlib import Path
from PIL import Image



def generate_centerLine(img_arr):
    binary = img_arr < 128  # invert image
    
    skeleton = skeletonize(binary) #returns a 2D boolean array (True = track skeleton, False = background)

    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    points = max(contours, key=cv2.contourArea)
    return points

def resample_points(skeleton_points):
    epsilon = 0.0005 * cv2.arcLength(skeleton_points, True) ###################### experiment with the multiplier value, 0.0004 ==> num of points = 88, 0.0003 ==> 112, 0.0005 ==> 81
    poly = cv2.approxPolyDP(skeleton_points, epsilon, True)
    print('number of sampled points: ',len(poly))
    poly = poly.reshape(-1, 2)

    return poly

def catmull_rom_spline(points):
    ALPHA = 0.5
    
    sample_per_segments = 20

    points = np.array(points, dtype=float)
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    points = np.vstack([points[-1], points, points[0], points[1]]) #ensure closed curve by adding the last point and the start and the first two points to the end
    num_points = len(points) # number of control points 
    result = []

    def tj(ti, pi, pj):
        dx, dy = pj - pi
        l = np.hypot(dx, dy)
        return ti + (l ** ALPHA)

    for i in range(1, num_points-2): #since catmull rom needs 4 points to generate 1 segment so the 2 control points are ignored, also because we added points so we need to ignore them
        P0 = points[i - 1]
        P1 = points[i]
        P2 = points[i + 1]
        P3 = points[i + 2]

    
        t0 = 0.0
        t1 = tj(t0, P0, P1)
        t2 = tj(t1, P1, P2)
        t3 = tj(t2, P2, P3)
        t = np.linspace(t1, t2,sample_per_segments, endpoint=False)


        A1 = (t1 - t)[:,None]/(t1 - t0) * P0 + (t - t0)[:,None]/(t1 - t0) * P1
        A2 = (t2 - t)[:,None]/(t2 - t1) * P1 + (t - t1)[:,None]/(t2 - t1) * P2
        A3 = (t3 - t)[:,None]/(t3 - t2) * P2 + (t - t2)[:,None]/(t3 - t2) * P3

        B1 = (t2 - t)[:,None]/(t2 - t0) * A1 + (t - t0)[:,None]/(t2 - t0) * A2
        B2 = (t3 - t)[:,None]/(t3 - t1) * A2 + (t - t1)[:,None]/(t3 - t1) * A3

        C  = (t2 - t)[:,None]/(t2 - t1) * B1 + (t - t1)[:,None]/(t2 - t1) * B2
        result.append(C)

    curve = np.vstack(result)
    curve = np.vstack([curve, curve[0]]) 
    return curve

def spline_properties(curve):

    diffs = np.diff(np.vstack([curve, curve[0]]), axis=0) #esnure return to start
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cumsum = np.concatenate(([0.0], np.cumsum(seg_lengths)))
    cumsum = cumsum[:-1]
    total_length = cumsum[-1]

    x = curve[:,0]
    y = curve[:,1]

    # use a small eps to avoid divide by zero
    eps = 1e-9

    dx_ds = np.gradient(x, cumsum + eps)
    dy_ds = np.gradient(y, cumsum + eps)

    speed = np.hypot(dx_ds, dy_ds) + eps


    tangent = np.column_stack((dx_ds / speed, dy_ds / speed))
    normal = np.column_stack((-tangent[:,1], tangent[:,0])) #90deg rotation

    # second derivatives d2x/ds2, d2y/ds2
    d2x_ds2 = np.gradient(dx_ds, cumsum + eps)
    d2y_ds2 = np.gradient(dy_ds, cumsum + eps)

    #curvature 
    num = np.abs(dx_ds * d2y_ds2 - dy_ds * d2x_ds2)
    denom = (dx_ds**2 + dy_ds**2) ** 1.5
    # avoid division by zero
    denom_safe = np.where(denom == 0, 1e-12, denom)
    curvature = num / denom_safe
    r = 1 / np.where(curvature == 0, 1e-12, curvature) #radius of curvature
    return {
        'cumsum': cumsum,
        'speed': speed,
        'tangent': tangent,
        'normal': normal,
        'curvature': curvature,
        'length': total_length,
        'radius': r
        }
    
def generate_mesh(curve, track_width, mesh_res, num_points_across, normals):
    
    offsets = np.linspace(-track_width/2, track_width/2, num_points_across)

    #normalise normals
    norms = np.linalg.norm(normals, axis=1)
    norms[norms == 0] = 1.0 #avoid 0 div
    normals = normals / norms[:,None]

    
    mesh = []
    for i, (curve_pt, normal_vec) in enumerate(zip(curve, normals)):
        if i % mesh_res == 0:
            mesh_row_pts = curve_pt + np.outer(offsets, normal_vec) #outer product of mesh_row and normal_vev
            mesh.append(mesh_row_pts)
    return np.array(mesh)

def b_spline(pts, sample_size):
    x = pts[:,0]
    y = pts[:,1]
    tck, _ = interpolate.splprep([x, y], s=0, per=True) #returns t = knots, c = control points, k= degree
    u_fine = np.linspace(0, 1, sample_size) #number of points to have on the radius
    x_fine, y_fine = interpolate.splev(u_fine, tck, der=0) #evaluates the spline for 'sample_size' evenly spaced distance values
    dx, dy = interpolate.splev(u_fine, tck, der=1)
    d2x, d2y = interpolate.splev(u_fine, tck, der=2)
    curvature = (dx*d2y - dy*d2x) / (dx**2 + dy**2)**1.5

    pts = np.column_stack([x_fine, y_fine])
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])
        x = pts[:,0]
        y = pts[:,1]
        tck, _ = interpolate.splprep([x, y], s=0, per=True) #returns t = knots, c = control points, k= degree
        u_fine = np.linspace(0, 1, sample_size) #number of points to have on the radius
        x_fine, y_fine = interpolate.splev(u_fine, tck, der=0) #evaluates the spline for 'sample_size' evenly spaced distance values
        dx, dy = interpolate.splev(u_fine, tck, der=1)
        d2x, d2y = interpolate.splev(u_fine, tck, der=2)
        curvature = (dx*d2y - dy*d2x) / (dx**2 + dy**2)**1.5

    return np.column_stack([x_fine, y_fine]), curvature


def random_points(mesh, num_pts_across, rangepercent, sample_size):
    rand_pts_idx = []
    rangeVal = rangepercent*num_pts_across
    step = ceil(len(mesh) / sample_size)

    num_pts_across -= 1 #to ensure not out of range
    current_idx = random.randint(0, num_pts_across)

    mean = 0
    for i in range(0, len(mesh), step):
        target_idx = (num_pts_across) // 2 #the middle
        bias_factor = 0.01 # how much it pulls towards the center
        steered_center = (current_idx * (1 - bias_factor)) + (target_idx * bias_factor)
        
        start = max(0, int(steered_center - rangeVal))
        end = min(num_pts_across, int(steered_center + rangeVal)) #ensure not out of range
        
        mean += start
        current_idx = random.randint(start, end)
        rand_pts_idx.append(current_idx)
    
    rand_pts_idx[-1] = rand_pts_idx[0] #ensure closed loop
    for i in range(len(rand_pts_idx)-2, -1, -1):
        idx_old = rand_pts_idx[i+1]
        idx_new = rand_pts_idx[i]

        if abs(idx_new - idx_old) > rangeVal:
            start = max(0, int(idx_old - rangeVal))
            end = min(len(mesh[i]) -1,int(idx_old + rangeVal))
            rand_pts_idx[i] = random.randint(start, end)
    
    rand_pts = []
    for i, row in enumerate(range(0, len(mesh), step)):
        actual_pt_idx = rand_pts_idx[i]
        rand_pts.append(mesh[row][actual_pt_idx])

    return np.array(rand_pts)


def plot_boundaries(mesh, curve):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')
    plt.plot(curve[:,0], curve[:,1], 'b-', label='Centripetal Catmull–Rom')
    plt.axis('equal')
    plt.legend()
    plt.show()

def plot_mesh(mesh, img_arr):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]
    for row in mesh:
        plt.plot(row[:,0], row[:,1], 'k-')

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')
    
    plt.imshow(img_arr, cmap='gray')
    plt.axis('equal')
    plt.show()

def plot_everything(mesh, center_line, approx, rand_bsp, random_pts):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]
    for row in mesh:
        plt.plot(row[:,0], row[:,1], 'k-')

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')
    plt.plot(center_line[:,0], center_line[:,1], 'b-', label='Center line')
    plt.plot(approx[:,0], approx[:,1], 'ro-', label='Control Points')
    #plt.imshow(img_arr, cmap='gray')
    plt.plot(random_pts[:,0], random_pts[:,1], 'go-', label='random Points')
    plt.plot(rand_bsp[:,0], rand_bsp[:,1], 'p-', label='b-spline')

    plt.axis('equal')
    plt.show()


def plot_skeleton(pts, binary):
    num_pts = len(pts)
    # simple gradient: blue → red
    colors = np.zeros((num_pts, 3), dtype=np.uint8)
    for i in range(1, num_pts):
        t = i / (num_pts - 1)
        colors[i] = [
            int(255 * (1 - t)),  # blue decreases
            0,
            int(255 * t)         # red increases
        ]

    # --- Step 5: Draw the skeleton with color gradient ---
    vis = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for (x, y), c in zip(pts, colors):
        cv2.circle(vis, (x, y), 1, c.tolist(), -1)

    cv2.imshow("Skeleton (Gradient Order)", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_spline(curve, approx):
    #plt.imshow(img_arr, cmap='gray')
    plt.plot(approx[:,0], approx[:,1], 'ro-', label='Control Points')
    plt.plot(curve[:,0], curve[:,1], 'b-', label='Centripetal Catmull–Rom')
    plt.legend()
    plt.axis('equal')
    plt.show()

def plot_bspline(b_spline, sampledpts, mesh=None, curvature=None, cmap='plasma'):
    fig, ax = plt.subplots(figsize=(8, 8))

    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')

    # ----- curvature-based coloring -----

    # Ensure curvature has same length as spline
    curvature = np.asarray(curvature)
    curvature = np.abs(curvature)  # magnitude only
    curv_min, curv_max = np.min(curvature), np.max(curvature)
    curvature_norm = (curvature - curv_min) / (curv_max - curv_min + 1e-12)

    # Create colored line segments
    points = b_spline.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1))
    lc.set_array(curvature_norm)
    lc.set_linewidth(2.5)
    ax.add_collection(lc)

    # Add colorbar
    cbar = plt.colorbar(lc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Curvature (normalized)")




    # ----- formatting -----
    ax.set_aspect('equal', 'box')
    ax.set_title("B-spline Racing Line with Curvature Heatmap")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.5)

    plt.show()

def plot_approx(approx, binary):

    vis = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for pt in approx:
        cv2.circle(vis, tuple(pt), 2, (0, 0, 255), -1)

    cv2.imshow("Approximated Polygon", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def plot_img(img_arr):
    plt.imshow(img_arr, cmap='gray')
    plt.show()

def return_Img_Arr(track_name):
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]

    ASSETS_DIR = _PROJECT_ROOT / "assets" / "tracks"
    ORDERS_DIR = _PROJECT_ROOT / "orders" / track_name
    ORDERS_DIR.mkdir(exist_ok=True)
    filepath = ASSETS_DIR / f"{track_name}.png"

    img = Image.open(filepath).convert('L') # ensure grayscale
    return np.asarray(img)

def main(track_name, real_properties, num_points_across=50, mesh_res=1, rangepercent=0.05):
    #loading image
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]

    ASSETS_DIR = _PROJECT_ROOT / "assets" / "tracks"
    ORDERS_DIR = _PROJECT_ROOT / "orders" / track_name
    ORDERS_DIR.mkdir(exist_ok=True)
    filepath = ASSETS_DIR / f"{track_name}.png"

    img = Image.open(filepath).convert('RGBA')
    # Create white background image
    white_bg = Image.new('RGBA', img.size, (255, 255, 255, 255))
    # Composite image over white background
    img_composite = Image.alpha_composite(white_bg, img)
    img_gray = img_composite.convert('L') # ensure grayscale
    img_arr = np.asarray(img_gray)

    binary = img_arr < 128  # invert image

    #variables


    num_points_across= 50
    mesh_res = 4 #the bigger the number the lower the resolution
    
    rangepercent = 0.1 #the lower the number the lower the range --> less spiky curve between 0,1

    #running functions

    skeleton_points_3d = generate_centerLine(img_arr)
    skeleton_points = skeleton_points_3d[0] #this is a more useful version
    approx = resample_points(skeleton_points_3d) #cv2 requires it in this format for contours


    center_line = catmull_rom_spline(approx)
    center_line_properties = spline_properties(center_line)

    convertion = center_line_properties['length'] / real_properties[track_name]['real_track_length'] #num of pixels per meter
    track_width_pixels = convertion * real_properties[track_name]['real_track_width']
    print('num of pixels per meter: ', convertion)

    
    mesh = generate_mesh(center_line, track_width_pixels, mesh_res, num_points_across, center_line_properties['normal']) 

    
    random_pts = random_points(mesh, num_points_across, rangepercent, sample_size=1000)

    if not np.allclose(random_pts[0], random_pts[-1]):
        random_pts = np.vstack([random_pts, random_pts[0]]) #temp double double check it is a close loop

    rand_bsp, curvature = b_spline(random_pts, sample_size= 1000)
    radius = 1/abs(curvature)
    #print(repr(random_pts))
 

    #print('cv2s arclength feature vs arc-length params:', cv2.arcLength(center_line.astype(np.float32).reshape(-1,1,2), True), 'vs', center_line_properties['length'])

    

    #plot_img(img_arr)
    #plot_skeleton(skeleton_points, binary)
    #plot_approx(approx, binary)
    #plot_spline(center_line, approx)
    #plot_spline(rand_bsp, random_pts)
    #plot_bspline(rand_bsp, random_pts, mesh, curvature)
    #plot_boundaries(mesh, center_line)
    #plot_mesh(mesh, img_arr)
    #plot_everything(mesh, center_line, approx, rand_bsp, random_pts)
    return approx, center_line, center_line_properties, mesh



if __name__ == "__main__":
    real_properties = {
    'silverstone': {
        'real_track_length': 5891, #meters
        'real_track_width': 20 #meters
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
    main('silverstone', real_properties)