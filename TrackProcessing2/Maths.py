import numpy as np

def normalized_cumulative(pts):
    distances = np.hypot(np.diff(pts[:,0]), np.diff(pts[:,1]))
    sum = np.concatenate([[0.0], np.cumsum(distances)])   # length == len(pts)
    if sum[-1] == 0:
        return sum, sum #when end and start are same point
    return sum, sum / sum[-1] #sum and normalized sum

def unit_tangets(pts):
    m = np.gradient(pts, axis = 0)
    T = m / (np.linalg.norm(m, axis=1, keepdims=True)) #norm helps calculate the magnitude
    N = np.column_stack([-T[:,1], T[:,0]]) #perpendicular vectors, also normalised
    return m, T, N

def line_projection_intersection(p0, N, pts): #takes in the 'mid point', Unit tangent(gradient), and the curve
    #does pts lie on the line p0 + t*T (for some t)
    corresponding_pts = []
    #l1 is the normal of the the mid curve of the track at p0
    for i, p in enumerate(p0):
        best_dist = np.inf
        best_pt = None
        print('n:', N[i])
        print('p:', p)
        k = N[i][0]*p[0] + N[i][1]*p[1]
        print('k:', k)
        for pt in pts:
            if N[i][0] == 0: #vertical line
                pass #cba rn
            elif N[i][1] == 0: #horizontal line
                pass #cba rn
            else:
                
                c = N[i][0]*pt[0] + N[i][1]*pt[1]
                print('c:', c)
                if k == c: #point lies on the line
                    len = np.hypot(pt[0]-p[0], pt[1]-p[1])
                    if len < best_dist:
                        best_dist = len
                        best_pt = pt
            corresponding_pts.append(best_pt)
    return corresponding_pts

def line_intersection_optimized(mid_pts, normals, pts):
    x_b, y_b = pts[:,0], pts[:,1] #boundary pts
    bx1, bx2 = x_b[:-1], x_b[1:] #bx1 = start x of segment, bx2 = end x of segment
    by1, by2 = y_b[:-1], y_b[1:]
    vsegx = bx2 - bx1 # segment direction vector x
    vsegy = by2 - by1

    intersections = []
    for i, (p, N) in enumerate(zip(mid_pts, normals)): #matches i with point with normals
         # line constant
        k = N[0]*p[0] + N[1]*p[1]


#example usage
p0 = np.array([[0,3],[4,7], [8, 9]])
curve = np.array([[1,2],[2,2],[3,3],[4,5],[5,8]])
#print('normalized_cumulative:', normalized_cumulative(p0))
m, T, N = unit_tangets(p0)
print('gradient vector: ',m, '\n Unit tangent vector:',T, '\n Unit Normal:', N )
print('intersections', line_projection_intersection(p0, N, curve))
