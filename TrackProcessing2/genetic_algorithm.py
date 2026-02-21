import pygame as pg
from pygame.locals import *
import numpy as np
from math import hypot
import random
import time
import os
from TrackProcessing2.FindTrackTime import main as find_track_time
from TrackProcessing2.FindTrackTime import plot_velocity_colored_line
import TrackProcessing2.generateSpline as generateSpline
from TrackProcessing2.config import config, real_properties
# from FindTrackTime import main as find_track_time
# from FindTrackTime import plot_velocity_colored_line
# import generateSpline as generateSpline
# from config import config, real_properties, default_variables
from enum import Enum
import matplotlib.pyplot as plt



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

def initialize_population(pop_size, track_name, variables, mesh, center_line_properties):
    population = []
    for i in range(pop_size):
        #print(f'\r individual number: {i}', end='')
        rand_bsp, radius = create_random_bsp(mesh, track_name, center_line_properties)
        vel, t = find_track_time(rand_bsp, radius, pixels_per_meter, variables)
        population.append([rand_bsp, t, vel])
    return population

def evaluate_population(population):
    population.sort(key=lambda x: x[1][-1]) #sort by time
    return population

def select_parents(population):
    n = len(population)
    # weights = np.linspace(1, 0, n) # Square it to favor the top individuals heavily
    
    # probs = weights / np.sum(weights)
    # print(probs)

    times = np.array([individiual[1][-1] for individiual in population]) #get times
    fitness = 1.0 / (times) if times.all() > 0 else np.ones_like(times) #avoid div by zero
    probs = fitness / np.sum(fitness)
    parent1 = population[np.random.choice(n, p=probs)]
    parent2 = population[np.random.choice(n, p=probs)]
    return parent1, parent2

def crossover(parent1, parent2):
    SECTOR_LENGTH = config["SECTOR_LENGTH"] #points per sector

    p1_coords, _, p1_vels = parent1
    p2_coords, _, p2_vels = parent2
    child_coords = np.zeros_like(p1_coords)

    # padding = SECTOR_LENGTH
    # p1_padded = np.concatenate([p1_vels[-padding:], p1_vels, p1_vels[:padding]]) #add padding to make convolution circular
    # p2_padded = np.concatenate([p2_vels[-padding:], p2_vels, p2_vels[:padding]])
 
    # p1_smooth = np.convolve(p1_padded, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same') #use convolve - moving average allows to smooth out velocities over each sector
    # p2_smooth = np.convolve(p2_padded, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same')

    # p1_smooth_vel = p1_smooth[padding:-padding]#remove padding
    # p2_smooth_vel = p2_smooth[padding:-padding]

    p1_smooth_vel = np.convolve(p1_vels, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same') #use convolve - moving average allows to smooth out velocities over each sector
    p2_smooth_vel = np.convolve(p2_vels, np.ones(SECTOR_LENGTH)/SECTOR_LENGTH, mode='same')

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
            

            midpoint = (individual[prev_idx] + individual[next_idx]) / 2
            new_coords[i] = (1 - smoothing_factor) * individual[i] + smoothing_factor * midpoint #lerp
            
        return new_coords
    def mutate_nudge(individual, target, nudging_factor):
        n = len(individual)

        window_size = int(n*config["window_factor"])  #% of points to nudge
        
        displacement = np.random.uniform(-nudging_factor, nudging_factor, size=2)
        
        for i in range(-window_size, window_size):
            #Apply Gaussian blur to create smooth nudge
            sigma = window_size / 2
            weight = np.exp(-0.5 * (i / sigma)**2)
            
            curr_idx = (target + i) % n
            individual[curr_idx] += displacement * weight
            
        return individual

    if generation < total_generations * config["smooth_to_nudge_factor"]:
        return mutate_smooth(individual, smoothing_factor)
    else:
        threshold = np.percentile(radius_arr, 10) #pick top 10% smallest radii --> sharpest turns
        corner_indices = np.where(radius_arr <= threshold)[0] #find corners based on radius
        
        if len(corner_indices) > 0:
            target = np.random.choice(corner_indices) #pick a random corner 
            return mutate_nudge(individual, target, nudging_factor)
            
    return individual

def init_track(variables):
    return generateSpline.main(variables)

def create_random_bsp(mesh, track_name, center_line_properties):
    random_pts = generateSpline.random_points(mesh)
    if not np.allclose(random_pts[0], random_pts[-1]):
        random_pts = np.vstack([random_pts, random_pts[0]]) #temp double double check it is a close loop
    rand_bsp, curvature = generateSpline.b_spline(random_pts)
    if not np.allclose(rand_bsp[0], rand_bsp[-1]):
        rand_bsp = np.vstack([rand_bsp, rand_bsp[0]])


    pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']

    # compute radius (1/curvature), skip zero curvature to avoid div by zero
    #radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]
    radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in curvature]
    radius =  np.array(radius) / pixels_per_meter
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
    

def main(variables):
    global mesh
    global center_line_properties
    global pixels_per_meter

    track_name = variables["track"]
    start_time = time.time()
    center_line_ctrpts, center_line, center_line_properties, mesh = init_track(variables)

    best_time_arr = []
    
    print('Initializing population...')

    pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']

    pop_size = config["pop_size"]
    print('Initializing population...')
    population = initialize_population(pop_size, track_name, variables, mesh, center_line_properties)
    elite_rate = config['elite_rate']
    mut_rate = config['mut_rate']
    smoothing_factor = config['smoothing_factor']
    nudging_factor = config['nudging_factor']
    
    total_generations = config['total_generations']
    print('\nStarting Genetic Algorithm...')
    for generation in range(total_generations):
        population = evaluate_population(population)
        elites = population[:int(pop_size * elite_rate)]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)
            
            props = generateSpline.spline_properties(child)
            radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]
            radius = np.array(radius) / pixels_per_meter
            if random.uniform(0, 1) < mut_rate:
                child = mutate(child, radius, smoothing_factor, nudging_factor, generation, total_generations)
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
        if np.var(best_time_arr[-5:]) < 0.05 and generation > 15:
            break #stop early if varience is low

    t = time.time() - start_time
    print(f"time taken: {int(t//60)}: {t%60:.3f}")
    try:
        best = min(population, key=lambda x: x[1][-1])  # returns tuple (spline, time, velocity)
        best_spline = best[0]
        best_time = best[1]
        best_vels = best[2]
    except IndexError: print(best)

    #plot_just_best_line(best_spline, mesh)
    #plot_velocity_colored_line(best_spline, best_vels)
    #plot_times(best_time_arr)
    return best_spline, best_time, best_vels, mesh
    


# if __name__ == '__main__':
#     variables = default_variables
#     main(variables)