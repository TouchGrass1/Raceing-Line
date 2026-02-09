import numpy as np
import random
import time
from TrackProcessing2.FindTrackTime import main as find_track_time
from TrackProcessing2.FindTrackTime import plot_velocity_colored_line
import TrackProcessing2.generateSpline as generateSpline
from enum import Enum
import matplotlib.pyplot as plt
from TrackProcessing2.config import config, real_properties



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

def initialize_population(pop_size, variables):
    population = []
    for i in range(pop_size):
        #print(f'\r individual number: {i}', end='')
        rand_bsp, radius = create_random_bsp(mesh)
        vel, t = find_track_time(rand_bsp, radius, pixels_per_meter, variables)
        population.append([rand_bsp, t, vel])
    return population

def evaluate_population(population):
    population.sort(key=lambda x: x[1][-1]) #sort by time
    return population

def select_parents(population, num_children):
    

    n = len(population)

    weights = np.linspace(1, 0, n)**2 # Square it to favor the top individuals heavily
    probs = weights / np.sum(weights)

    p1_idx = np.random.choice(n, size=num_children, p=probs[:n], replace=True)
    p2_idx = np.random.choice(n, size=num_children, p=probs[:n], replace=True)
    return p1_idx, p2_idx

def crossover(parent1, parent2):
    SECTOR_LENGTH = 800 #points per sector

    p1_coords, _, p1_vels = parent1
    p2_coords, _, p2_vels = parent2
    child_coords = np.zeros_like(p1_coords)

    padding = SECTOR_LENGTH
    p1_padded = np.concatenate([p1_vels[-padding:], p1_vels, p1_vels[:padding]]) #add padding to make convolution circular
    p2_padded = np.concatenate([p2_vels[-padding:], p2_vels, p2_vels[:padding]])
 
    p1_smooth = np.convolve(p1_padded, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same') #use convolve - moving average allows to smooth out velocities over each sector
    p2_smooth = np.convolve(p2_padded, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same')

    p1_smooth_vel = p1_smooth[padding:-padding]#remove padding
    p2_smooth_vel = p2_smooth[padding:-padding]

    choices = (p2_smooth_vel > p1_smooth_vel).astype(float) #compare speeds from each sector
    smooth_choices = np.convolve(choices, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same') #smooth out the choices to make the linear interpolation less spiky

    for i in range(len(child_coords)):
        weight = smooth_choices[i]
        child_coords[i] = (1 - weight) * p1_coords[i] + weight * p2_coords[i] #LERP

    return child_coords

def mutate(individual, radius_arr, smoothing_factor, nudging_factor, generation, total_generations):

    def mutate_smooth(individual, smoothing_factor):
        new_coords = np.copy(individual) #copy 
        n = len(individual)

        for i in range(n):
            prev_idx = (i - 1) % n #get surrounding nodes 
            next_idx = (i + 1) % n
            

            midpoint = (individual[prev_idx] + individual[next_idx]) / 2.0
            new_coords[i] = (1 - smoothing_factor) * individual[i] + smoothing_factor * midpoint #lerp
            
        return new_coords
    def mutate_nudge(individual, target, nudging_factor):
        n = len(individual)

        window_size = n // 10  #% of points to nudge
        
        displacement = np.random.uniform(-nudging_factor, nudging_factor, size=2) #random displacement vector
        
        #pre calculate gaussian weights
        indices = np.arange(-window_size, window_size)
        sigma = window_size / 2
        weights = np.exp(-0.5 * (indices / sigma)**2)

        #apply gaussian smoothing
        weighted_disp = displacement * weights[:, np.newaxis]
        target_idx = (target + indices) % n
        individual[target_idx] += weighted_disp
        
        return individual

    if generation < total_generations * 0.2: #mutate smooth
        prev_pts = np.roll(individual, 1, axis=0)
        next_pts = np.roll(individual, -1, axis=0)
        midpoints = (prev_pts + next_pts) * 0.5

        return (1 - smoothing_factor) * individual + smoothing_factor * midpoints #LERP
    else:
        threshold = np.percentile(radius_arr, 10) #pick top 10% smallest radii --> sharpest turns
        corner_indices = np.where(radius_arr <= threshold)[0] #find corners based on radius
        
        if len(corner_indices) > 0:
            target = np.random.choice(corner_indices) #pick a random corner 
            return mutate_nudge(individual, target, nudging_factor)
            
    return individual

def init_track(track_name):
    return generateSpline.main(track_name, real_properties)

def create_random_bsp(mesh):
    sample_size= config["sample_size"]
    random_pts = generateSpline.random_points(mesh, num_pts_across=50, rangepercent=0.02, sample_size=sample_size)
    if not np.allclose(random_pts[0], random_pts[-1]):
        random_pts = np.vstack([random_pts, random_pts[0]]) #temp double double check it is a close loop
    rand_bsp, curvature = generateSpline.b_spline(random_pts, sample_size)
    if not np.allclose(rand_bsp[0], rand_bsp[-1]):
        rand_bsp = np.vstack([rand_bsp, rand_bsp[0]])

    abs_curv = np.abs(curvature)
    valid_mask = abs_curv > 1e-6
    
    radius = np.full_like(curvature, np.inf)
    radius[valid_mask] = 1.0 / abs_curv[valid_mask]

    radius =  radius/ pixels_per_meter
    return rand_bsp, radius

def plot_just_best_line(spline, mesh):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')


    plt.plot(spline[:,0], spline[:,1], 'b-')
    plt.axis('equal')
    plt.show()

def plot_times(best_time_arr):
    plt.plot(best_time_arr, marker='o')
    plt.title('Best Time per Generation')
    plt.xlabel('Generation')
    plt.ylabel('Best Time (s)')
    plt.grid()
    plt.show()

def plotVelocity(vel):
    x = np.arange(len(vel))
    plt.plot(x, vel)
    plt.show()   

def main(variables, population, pop_size):
    
    start_time = time.time()    
    best_time_arr = []

    #50
    print('Initializing population...')
    
    elite_rate = config['elite_rate']
    mut_rate = config['mut_rate']
    smoothing_factor = config['smoothing_factor']
    nudging_factor = config['nudging_factor']

    

    
    total_generations = config['total_generations'] #50
    num_elites = int(pop_size * elite_rate)
    num_children = pop_size - num_elites
    print('\nStarting Genetic Algorithm...')

    for generation in range(total_generations):
        population = evaluate_population(population)
        elites = population[:num_elites]

        progress = generation / total_generations
        current_mut_rate = max(0.05, mut_rate * (1 - progress))
        current_nudge = max(0.5, nudging_factor * (1 - progress))

        new_population = elites.copy()
        p1_idx, p2_idx = select_parents(population, num_children)
        for i in range(num_children):
            parent1 = population[p1_idx[i]]
            parent2 = population[p2_idx[i]]
            child = crossover(parent1, parent2)
            
            props = generateSpline.spline_properties(child)
            radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]
            radius = np.array(radius) / pixels_per_meter

            if random.uniform(0, 1) < current_mut_rate:
                child = mutate(child, radius, smoothing_factor, current_nudge, generation, total_generations)
                props = generateSpline.spline_properties(child) #reapply properties after mutation
                radius = np.array([1/abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]) / pixels_per_meter
            vel, t = find_track_time(child, radius, pixels_per_meter, variables)
            new_population.append((child, t, vel))

        
        population = new_population

        best_time = min([p[1][-1] for p in population])
        best_time_arr.append(best_time)
        plt.close('all')
        print(f"Gen {generation}: Best time = {best_time:.3f}s")
        print('Varience: ', np.var(best_time_arr[-5:]))
        print('-----------------------')
        if np.var(best_time_arr[-5:]) < 0.05 and generation > config["min_generations"]:
            break #stop early if varience is low

    t = time.time() - start_time
    print(f"time taken: {int(t//60)}: {t%60:.3f}")
    try:
        best = min(population, key=lambda x: x[1][-1])  # returns tuple (spline, time, velocity)
    except IndexError: print(best)


    return best

def repeat(variables):
    global mesh
    global center_line_properties
    global pixels_per_meter

    start_time = time.time()
    n = config['total_repeats'] #number of repeats

    #Initialize track
    center_line_ctrpts, center_line, center_line_properties, mesh = init_track(variables['track'])
    pixels_per_meter = center_line_properties['length'] / real_properties[variables['track']]['real_track_length']
    results_arr = []

    #repeat genetic algorithm with random initial populations
    for i in range(n):
        population = initialize_population(config['pop_size'], variables)
        best = main(variables, population, config['pop_size'])
        results_arr.append(best)

    #repeat genetic algorithm with population of the best from each run
    best = main(variables, results_arr, n)


    t = time.time() - start_time
    print(f"time taken: {int(t//60)}: {t%60:.3f}")
    try:
        best = min(population, key=lambda x: x[1][-1])  # returns tuple (spline, time, velocity)
        best_spline = best[0]
        best_time = best[1]
        best_vels = best[2]
    except IndexError: print(best)

    plot_just_best_line(best_spline, mesh)
    plot_velocity_colored_line(best_spline, best_vels)
    return best_spline, best_time, best_vels, mesh

    




# if __name__ == '__main__': 
#     main(track_name='silverstone')