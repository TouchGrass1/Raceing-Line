import generateSpline
import pygame as pg
import time
from enum import Enum
from pygame.locals import *

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

#PYGAME STUFF
pg.init()
start_time = time.time()

screen = pg.display.set_mode(flags=pg.FULLSCREEN)
#screen = pg.display.set_mode((720,640))
screen_shape = screen.get_size()
pg.display.set_caption('Racing Lines')

# Background surface (static)
background = pg.Surface(screen.get_size())
background = background.convert()

background.fill(colour_palette['BG_GREY'].value)
screen.blit(background, (0, 0))




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
    for event in pg.event.get():
        if event.type == QUIT:
            run = False
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            run = False
        if event.type == MOUSEBUTTONDOWN:
            print(pg.mouse.get_pos())
    

    #BASE
    screen.blit(background, (0, 0))
    pg.draw.lines(screen,colour_palette['ORANGE'].value, True, left_boundary)
    pg.draw.lines(screen,colour_palette['ORANGE'].value, True, right_boundary)
    pg.draw.lines(screen,colour_palette['RED'].value, True, center_line_ctrpts)
    pg.draw.lines(screen,colour_palette['BLUE'].value, True, random_pts)


    #car
    pos = random_pts[i]
    pg.draw.circle(screen, colour_palette['WHITE'].value, pos, 2)

    time.sleep(0.1)
    pg.display.flip()

    if i == len(random_pts) -1:
        i = 0
    else: i+=1

