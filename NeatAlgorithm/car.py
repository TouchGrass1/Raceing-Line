import numpy as np
from math import sin, cos, atan2, sqrt

class CarState:
    def __init__(self, x=0.0, y=0.0, heading=0.0, speed=0.0):
        self.x = x
        self.y = y
        self.heading = heading  # radians, 0 = +x
        self.speed = speed

class SimpleCar:
    def __init__(self, start_pos, start_heading=0.0):
        self.state = CarState(x=start_pos[0], y=start_pos[1], heading=start_heading, speed=0.0)
        # physical limits -- for now it is fixed but willchange
        self.max_steer = 0.6  # radians per second (not wheel angle)
        self.max_accel = 4.0  
        self.max_speed = 80.0
        self.length = 2.5  # wheelbase approx

        # simulation variables
        self.lap_complete = False
        self.off_track = False
        self.collided = False
        self.track_progress_idx = 0
    
    def update(self, steering_cmd, throttle_cmd, dt, track_utils):
        # steering and throttle is between -1 to 1
        steer = max(-1.0, min(1.0, steering_cmd)) * self.max_steer
        accel = max(-1.0, min(1.0, throttle_cmd)) * self.max_accel

        if abs(self.state.speed) > 0.001:
            turning_radius = max(0.0001, self.length / (steer + 1e-6))
            ang_vel = self.state.speed / turning_radius #angular velocity
        else:
            ang_vel = 0.0
        
        self.state.heading += ang_vel * dt
        self.state.speed += accel * dt

        #clamp
        self.state.speed = max(0.0, min(self.state.speed, self.max_speed))

        self.state.x += self.state.speed * cos(self.state.heading) * dt
        self.state.y += self.state.speed * sin(self.state.heading) * dt

        #check if out of bounds
        idx = track_utils['nearest_index_on_line']((self.state.x, self.state.y)) #finds nearest normal/value on center line
        self.track_progress_idx = idx
        half_width = 0.5*track_utils['width']

        dist_to_center = np.linalg.norm(np.array([self.state.x, self.state.y]) - track_utils['center_line'][idx])
        if dist_to_center > half_width + self.length:  # tolerance
            self.off_track = True
        
        #check for completion
        if track_utils['progress_fraction'](self.track_progress_idx) > 0.99:
            self.lap_complete = True