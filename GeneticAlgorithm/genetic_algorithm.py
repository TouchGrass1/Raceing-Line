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
    for _ in range(pop_size):
        rand_bsp, radius = create_random_bsp(mesh, track_name, center_line_properties)
        vel, t = find_track_time(rand_bsp, radius, pixels_per_meter)
        population.append([rand_bsp, t])
    return population

def evaluate_population(population):
    population.sort(key=lambda x: x[1]) #sort by time
    return population

def select_parents(population):
    times = np.array([ind[1] for ind in population])
    fitness = 1.0 / (times + 1e-6)
    probs = fitness / np.sum(fitness)

    parent1 = population[np.random.choice(len(population), p=probs)][0]
    parent2 = population[np.random.choice(len(population), p=probs)][0]
    return parent1, parent2

def crossover(parent1, parent2):
    split = random.uniform(0.25, 0.75)
    idx = int(len(parent1) * split)
    child = np.vstack([parent1[:idx], parent2[idx:], parent1[0]])
    return child

def mutate(individual, mutation_rate, mutation_strength):
    return

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

def plot_just_spline(spline):
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

    pop_size = 100
    population = initialize_population(pop_size)
    elite_rate = 0.1
    crossover_rate = 0.8
    mut_rate = 0.1
    mut_strength = 0.05
    
    generations = 50

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

        best_time = min([p[1] for p in population])
        print(f"Gen {generation}: Best time = {best_time:.3f}s")

    print('time taken: ', time.time() - start_time)
    try:
        best = min(population, key=lambda x: x[1])  # returns tuple (spline, time, velocity)
        best_spline = best[0]
        best_time = best[1]
        best_vels = best[2]
    except IndexError: print(best)

    plot_just_spline(best_spline)
    plot_velocity_colored_line(best_spline, best_vels)


if __name__ == '__main__': main()