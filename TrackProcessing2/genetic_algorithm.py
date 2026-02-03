import pygame as pg
from pygame.locals import *
import numpy as np
from math import hypot
import random
import time
import os
from FindTrackTime import main as find_track_time
from FindTrackTime import plot_velocity_colored_line
import generateSpline
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

def initialize_population(pop_size):
    population = []
    for i in range(pop_size):
        print(f'\r individual number: {i}', end='')
        rand_bsp, radius = create_random_bsp(mesh, track_name, center_line_properties)
        vel, t = find_track_time(rand_bsp, radius, pixels_per_meter)
        population.append([rand_bsp, t, vel])
    return population

def evaluate_population(population):
    population.sort(key=lambda x: x[1][-1]) #sort by time
    return population

def select_parents(population):
    times = np.array([individiual[1][-1] for individiual in population]) #get times
    fitness = 1.0 / (times) if times.all() > 0 else np.ones_like(times) #avoid div by zero
    probs = fitness / np.sum(fitness)

    parent1 = population[np.random.choice(len(population), p=probs)]
    parent2 = population[np.random.choice(len(population), p=probs)]
    return parent1, parent2

def crossover(parent1, parent2):
    SECTOR_LENGTH = 100 #points per sector

    p1_coords, _, p1_vels = parent1
    p2_coords, _, p2_vels = parent2
    child_coords = np.zeros_like(p1_coords)

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
        return
    def mutate_nudge(individual, nudging_factor):
        return

    if generation < total_generations * 0.2:
        return mutate_smooth(individual, smoothing_factor)
    else:
        threshold = np.percentile(radius_arr, 10) #pick top 10% smallest radii --> sharpest turns
        corner_indices = np.where(radius_arr <= threshold)[0] #find corners based on radius
        
        if len(corner_indices) > 0:
            target_idx = np.random.choice(corner_indices)
            return mutate_nudge(individual, target_idx, nudging_factor)
            
    return individual

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

def plot_just_best_line(spline, mesh):
    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]

    plt.plot(left_boundary[:,0], left_boundary[:,1], 'r-', label='Left Boundary')
    plt.plot(right_boundary[:,0], right_boundary[:,1], 'g-', label='Right Boundary')


    plt.plot(spline[:,0], spline[:,1], 'b-')
    plt.axis('equal')
    plt.show()


    

def main():
    global mesh
    global center_line_properties
    global pixels_per_meter

    
    start_time = time.time()
    center_line_ctrpts, center_line, center_line_properties, mesh = init_track()


    pixels_per_meter = center_line_properties['length'] / real_properties[track_name]['real_track_length']

    pop_size = 25 #100
    print('Initializing population...')
    population = initialize_population(pop_size)
    elite_rate = 0.1
    crossover_rate = 0.8
    mut_rate = 0.1
    mut_strength = 0.05
    
    generations = 5 #50
    print('\nStarting Genetic Algorithm...')
    for generation in range(generations):
        population = evaluate_population(population)
        elites = population[:int(pop_size * elite_rate)]

        new_population = elites.copy()
        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population)
            child = crossover(parent1, parent2)

            props = generateSpline.spline_properties(child)
            radius = [1 / abs(c) if abs(c) > 1e-6 else np.inf for c in props["curvature"]]
            radius = np.array(radius) / pixels_per_meter
            vel, t = find_track_time(child, radius, pixels_per_meter)
            new_population.append((child, t, vel))

        
        population = new_population

        best_time = min([p[1][-1] for p in population])
        print(f"Gen {generation}: Best time = {best_time:.3f}s")

    print('time taken: ', time.time() - start_time)
    try:
        best = min(population, key=lambda x: x[1][-1])  # returns tuple (spline, time, velocity)
        best_spline = best[0]
        best_time = best[1]
        best_vels = best[2]
    except IndexError: print(best)

    plot_just_best_line(best_spline, mesh)
    plot_velocity_colored_line(best_spline, best_vels)


if __name__ == '__main__': main()