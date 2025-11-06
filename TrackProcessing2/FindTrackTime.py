import generateSpline
import pygame as pg
import time
from enum import Enum
from pygame.locals import *
from Physics import *

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
    }
}



def uppdateVelocity(vel, mu, mass):
    downforce = downforceEquation(vel)
    maxLateralForce = maxLateralForceEquation(mu, mass, downforce)
    lateralForce = lateralForceEquation(mass, velocity, radius)
    if lateralForce > maxLateralForce:
        





def main():
    #SPLINE STUFF

    track_list = ["monza", "silverstone", "qatar"]
    track_name = track_list[1]

    center_line_ctrpts, center_line, center_line_properties, mesh = generateSpline.main(track_name, real_properties)

    random_pts = generateSpline.random_points(mesh, num_pts_across=50, rangepercent=0.1)

    rand_bsp, curvature = generateSpline.b_spline(random_pts, sample_size= 5000)
    radius = 1/abs(curvature)

    left_boundary = mesh[:,0,:]
    right_boundary = mesh[:,-1,:]

    #CAR
    i = 0


    pg.display.flip()
    run = True
    while run:




