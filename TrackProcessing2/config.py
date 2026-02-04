config = {
    "SECTOR_LENGTH": 800, #points per sector for crossover 
    "num_pts_across":50, 
    "rangepercent":0.02, #the lower the number the lower the range --> less spiky curve between 0,1
    "sample_size": 1000, #number of random points to sample on the track for BSP generation
    "pop_size": 1, #50
    "elite_rate": 0.1, #percentage of top individuals to carry over to next generation
    "mut_rate": 0.4, #mutation rate
    "smoothing_factor": 0.1, #the higher the number the more smoothing
    "nudging_factor": 2, #the higher the number the more aggressive the nudge
    
    "total_generations": 1, #50 
    "min_generations": 15, #minimum number of generations to run before checking for convergence
    "num_points_across": 50, #number of points to sample across the width of the track
    "mesh_res": 1, #the bigger the number the lower the resolution
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