import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

from skimage.util import invert
from scipy.interpolate import CubicSpline
from skimage.morphology import skeletonize

from geomdl import BSpline, utilities
from geomdl.visualization import VisMPL


from pathlib import Path
from PIL import Image



def generate_centerLine(img_arr):
    binary = img_arr < 128  # invert image
    
    skeleton = skeletonize(binary) #returns a 2D boolean array (True = track skeleton, False = background)

    y, x = np.nonzero(skeleton)  # get pixel coordinates of skeleton
    points = np.column_stack((x, y)).astype(np.int32)
    return points, skeleton

def resample_points(skeleton, binary):

    contours, _ = cv2.findContours(skeleton.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.0004 * cv2.arcLength(largest, True) ###################### experiment with the multiplier value, 0.0004 ==> num of points = 93, 0.0003 ==> 113, 0.0005 ==> 83
    poly = cv2.approxPolyDP(largest, epsilon, False)
    print('number of sampled points: ',len(poly))
    poly = poly.reshape(-1, 2)





    return poly

def catmull_rom_spline(points):
    ALPHA = 0.5
    num_points = len(points) + 2 # number of control points 

    points = np.array(points, dtype=float)
    points = np.vstack([points[-1], points, points[0], points[1]]) #ensure closed curve by adding the last point and the start and the first two points to the end

    result = []

    def tj(ti: float, pi: tuple, pj: tuple) -> float:
        dx, dy = pj - pi
        l = np.hypot(dx, dy)
        return ti + l ** ALPHA

    for i in range(1, num_points - 2): #since catmull rom needs 4 points to generate 1 segment so the 2 control points are ignored, also because we added points so we need to ignore them
        P0 = points[i - 1]
        P1 = points[i]
        P2 = points[i + 1]
        P3 = points[i + 2]

    
        t0 = 0.0
        t1 = tj(t0, P0, P1)
        t2 = tj(t1, P1, P2)
        t3 = tj(t2, P2, P3)
        t = np.linspace(t1, t2, num_points).reshape(num_points, 1)

        A1 = (t1 - t) / (t1 - t0) * P0 + (t - t0) / (t1 - t0) * P1
        A2 = (t2 - t) / (t2 - t1) * P1 + (t - t1) / (t2 - t1) * P2
        A3 = (t3 - t) / (t3 - t2) * P2 + (t - t2) / (t3 - t2) * P3

        B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
        B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3

        C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
        result.append(C)
    return np.vstack(result)

def spline_properties(curve):
    # arc length parameterization


    diffs = np.diff(curve, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    # add final segment from last->first
    last_len = np.linalg.norm(curve[0] - curve[-1])
    seg_lengths = np.concatenate([seg_lengths, [last_len]])

    N = curve.shape[0]
    cumsum = np.concatenate(([0.0], np.cumsum(seg_lengths)))[:N]




    total_length = cumsum[-1]
    x = curve[:,0]
    y = curve[:,1]

    dx_ds = np.gradient(x, cumsum)
    dy_ds = np.gradient(y, cumsum)

    speed = np.sqrt(dx_ds**2 + dy_ds**2)

    speed_safe = np.where(speed == 0, 0, speed) #remove 0s
    tangent = np.column_stack((dx_ds / speed_safe, dy_ds / speed_safe))
    normal = np.column_stack((-tangent[:,1], tangent[:,0])) #90deg rotation

    # second derivatives d2x/ds2, d2y/ds2
    d2x_ds2 = np.gradient(dx_ds, cumsum)
    d2y_ds2 = np.gradient(dy_ds, cumsum)

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
        'length': total_length
        }
    
def generate_mesh(curve, properties, real_properties):
    max_offset_distance = (properties['length'] * real_properties['real_track_width']) / real_properties['real_track_length']
    normals = properties['normal']
    mesh_res = 10 #num of points in each row
    mesh = []
    for i, (curve_pt, normal_vec) in enumerate(zip(curve, normals)):
        if i %20 == 0:
            mesh_row = np.linspace(-max_offset_distance/2, max_offset_distance/2, mesh_res)
            mesh_row_pts = curve_pt + np.outer(mesh_row, normal_vec) #outer product of mesh_row and normal_vev
            mesh.append(mesh_row_pts)
    return np.array(mesh)

def b_spline(pts):
    curve = BSpline.Curve()

    # Set degree
    curve.degree = 3

    # Set control points
    curve.ctrlpts = pts.tolist()

    # Auto-generate a knot vector
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    curve.sample_size = 5000
    curve.evaluate()
    return np.array(curve.evalpts)

def random_points(mesh):
    rand_pts = []
    for row in mesh:
        pt = random.choice(row)
        rand_pts.append(pt)
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

def plot_mesh(mesh):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]
    for row in mesh:
        plt.plot(row[:,0], row[:,1], 'k-')

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')
    

    plt.axis('equal')
    plt.show()

def plot_everything(mesh, center_line, approx):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]
    for row in mesh:
        plt.plot(row[:,0], row[:,1], 'k-')

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')
    plt.plot(center_line[:,0], center_line[:,1], 'b-', label='Center line')
    plt.plot(approx[:,0], approx[:,1], 'ro-', label='Control Points')

    plt.axis('equal')
    plt.show()


def plot_skeleton(pts, binary):
    num_pts = len(pts)
    # simple gradient: blue → red
    colors = np.zeros((num_pts, 3), dtype=np.uint8)
    for i in range(num_pts):
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

def plot_spline(curve, approx, img_arr):
    plt.imshow(img_arr, cmap='gray')
    plt.plot(approx[:,0], approx[:,1], 'ro-', label='Control Points')
    plt.plot(curve[:,0], curve[:,1], 'b-', label='Centripetal Catmull–Rom')
    plt.legend()
    plt.axis('equal')
    plt.show()
    plt.show()

def plot_approx(approx, binary):

    vis = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for pt in approx:
        cv2.circle(vis, tuple(pt), 2, (0, 0, 255), -1)

    cv2.imshow("Approximated Polygon", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    _PROJECT_ROOT = Path(__file__).resolve().parents[1]
    track_list = ["monza", "silverstone", "qatar"]
    track_name = track_list[1]

    ASSETS_DIR = _PROJECT_ROOT / "assets" / "tracks"
    ORDERS_DIR = _PROJECT_ROOT / "orders" / track_name
    ORDERS_DIR.mkdir(exist_ok=True)
    filepath = ASSETS_DIR / f"{track_name}.png"

    real_properties = {
        'silverstone': {
            'real_track_length': 5891, #meters
            'real_track_width': 12 #meters
        },
        'monza': {
            'real_track_length': 5793,
            'real_track_width': 10
        }
    }
    img = Image.open(filepath).convert("L")  # ensure grayscale
    img_arr = np.asarray(img)

    binary = img_arr < 128  # invert image

    points, skeleton = generate_centerLine(img_arr)
    print(cv2.arcLength(points, False))
    approx = resample_points(skeleton, binary)


    center_line = catmull_rom_spline(approx)

    mesh = generate_mesh(center_line, spline_properties(center_line), real_properties[track_name])

    random_pts = random_points(mesh)

    rand_bsp = b_spline(random_pts)


    #plot_skeleton(points, binary)
    #plot_approx(approx, binary)
    #plot_spline(cemter_line, approx, img_arr)
    #plot_spline(rand_bsp, random_pts, img_arr)
    #plot_boundaries(mesh, cemter_line)
    #plot_mesh(mesh)
    plot_everything(mesh, center_line, approx)


if __name__ == "__main__":
    main()