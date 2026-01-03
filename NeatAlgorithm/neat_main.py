import neat
import os
import numpy as np
import matplotlib.pyplot as plt
from math import atan2
from TrackSensor import make_ray_sensors, nearest_index_on_line, track_progress
from car import SimpleCar
import multiprocessing
import pickle
import time
import generateSpline_copy
from math import pi
import random

random.seed()
np.random.seed()

TRACK_NAME = 'silverstone' 
MAX_STEPS = 3000    # safety cap
DT = 1/60.0

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
    track_list = ["monza", "silverstone", "qatar", "90degturn"]
    track_name = track_list[1]
    center_line_ctrpts, center_line, center_line_properties, mesh = generateSpline_copy.main(track_name, real_properties)
    inner = mesh[:,0,:]
    outer = mesh[:,-1,:]
    return inner, outer, center_line, center_line_properties

global center
global center_line_properties
inner, outer, center, center_line_properties = init_track()
center = np.array(center)
inner = np.array(inner)
outer = np.array(outer)

def width(inner, outer):
    return np.linalg.norm(outer[0] - inner[0])

def nearest_index_on_line_func(pos):
    return nearest_index_on_line(pos, center)

def progress_fraction_func(pos):
    return track_progress(pos, center)

track_utils = {
    'center_line': center,
    'inner': inner,
    'outer': outer,
    'width': width(inner, outer),
    'nearest_index_on_line': nearest_index_on_line_func,
    'progress_fraction': progress_fraction_func
}


def gather_inputs(car, center, center_line_properties):
    max_dist = 300
    dists, angles = make_ray_sensors((car.state.x, car.state.y), car.state.heading, center, inner, outer, n_rays=9, fov=pi/2, max_dist=max_dist)
    # normalize distances by max_dist
    dists_norm = [d / max_dist for d in dists]

    progress = track_progress((car.state.x, car.state.y), center)
    idx = car.track_progress_idx
    # speed normalized (max_speed in car)
    speed_norm = car.state.speed / car.max_speed

    tangent_angle = atan2(center_line_properties['tangent'][idx][1], center_line_properties['tangent'][idx][0])
    heading_error = (car.state.heading - tangent_angle)
    while heading_error > pi: heading_error -= 2*pi #get value between 0-pi as it is circular
    while heading_error < -pi: heading_error += 2*pi
    heading_error_norm = heading_error / pi
    
    inputs = dists_norm + [progress, speed_norm, heading_error_norm] #Adjust NEAT config num_inputs accordingly if this changes.
    return np.array(inputs)


def eval_genome(genome, config):
    global center
    global center_line_properties
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    start_pos = center[0]
    car = SimpleCar(start_pos, start_heading=0.0)

    steer = 0

    genome_line = []
    genome_speeds = []
    t = 0.0
    crashed = False
    completed = False

    for step in range(MAX_STEPS):
        inputs = gather_inputs(car, center, center_line_properties)

        raw_out = net.activate(inputs.tolist())
        prevSteer = steer
        steer = raw_out[0]  # assume -1..1 via tanh
        throttle = raw_out[1]  # tanh -> -1..1

        car.update(steer, throttle, DT, track_utils)
        genome_line.append((car.state.x, car.state.y))
        genome_speeds.append(car.state.speed)
        t += DT

        if car.off_track:
            crashed = True
            break
        if car.lap_complete:
            completed = True
            break

    progress = track_progress((car.state.x, car.state.y), center)

    # Base reward: progress along track (0..1 scaled to 0..1000)
    fitness = progress * 1000

    # Reward speed *only when the car is on track*
    if not car.off_track:
        fitness += car.state.speed * 2.0     # moderate reward

    # Penalize excessive steering to encourage smoothness
    if step > 0:
        fitness -= abs(steer - prevSteer) * 5

    # Big reward for finishing a lap
    if car.lap_complete:
        lap_time = t
        fitness += 3000                      # completion bonus
        fitness += (3000 / lap_time)         # fast lap = more reward

    # Penalty for crashing
    if car.off_track:
        fitness *= 0.5
    
    genome.fitness_line = genome_line
    genome.fitness_speeds = genome_speeds
    genome.fitness_time = t

    if genome.key == 0 and step % 200 == 0:
        print("Progress:", progress, "Speed:", car.state.speed)
    print("Raw progress:", track_progress((car.state.x, car.state.y), center), '------------------------------------')

    return fitness


def run():
    generations = 350
    
    # load config
    config_path = 'NeatAlgorithm/neat-config.txt'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    

    p = neat.Population(config)
    # add reporters
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = p.run(pe.evaluate, generations)  # run for 50 generations (tweak)

    print("Winner fitness:", winner.fitness)

    best_line = getattr(winner, 'fitness_line', None)
    best_speeds = getattr(winner, 'fitness_speeds', None)

    if best_line is not None:
        xs = [p[0] for p in best_line]
        ys = [p[1] for p in best_line]
        plt.figure(figsize=(10,6))
        plt.plot(xs, ys, linewidth=1)
        plt.title('Winner Racing Line')
        plt.axis('equal')
        plt.show()

    if best_speeds is not None and best_line is not None:
        # color grade the line by speed (simple scatter)
        plt.figure(figsize=(10,6))
        sc = plt.scatter(xs, ys, c=best_speeds, s=6)
        plt.colorbar(sc, label='speed (m/s)')
        plt.title('Winner Racing Line colored by speed')
        plt.axis('equal')
        plt.show()

if __name__ == '__main__':
    start_time = time.time()
    run()
    print("Total time:", time.time() - start_time)
