config = {
    "SECTOR_LENGTH": 1000, #points per sector for crossover 
    "rangepercent":0.02, #the lower the number the lower the range --> less spiky curve between 0,1
    "sample_size": 1000, #number of random points to sample on the track for BSP generation
    "pop_size": 1, #50
    "elite_rate": 0.2, #percentage of top individuals to carry over to next generation
    "mut_rate": 0.3, #mutation rate
    "smoothing_factor": 0.1, #the higher the number the more smoothing
    "nudging_factor": 4, #the higher the number the more aggressive the nudge
    "window_factor": 0.1, #the size of the window during mutations
    "smooth_to_nudge_factor": 0.1, #when the swap between doing smoothing mutations to nudging mutations occur

    "total_generations": 50, #50 
    "total_repeats": 1, #20 number of times to repeat the whole GA 
    "min_generations": 15, #minimum number of generations to run before checking for convergence
    "num_points_across": 60, #number of points to sample across the width of the track
    "mesh_res": 1, #the bigger the number the lower the resolution
    "bias_factor": 0.8 # how much it pulls towards the center
}

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
    

default_variables = {
    'track': 'silverstone',
    'mass': 988,
    'weather': 'dry',
    'tyre': 'SOFT',
    'lapNo': 1,
    'elevation': 0,
    'temp': 20
}

variable_options = {
    'track': ['silverstone', 'monza', 'qatar', '90degturn', 'import'],
    'mass': [878, 988],
    'weather': ['dry', 'wet'],
    'tyre': ['SOFT', 'MEDIUM', 'HARD'],
    'lapNo': [1, 70],
    'elevation': [-28, 2200],
    'temp': [0, 45],
}
