import pygame as pg
from pygame.locals import *
import numpy as np
from math import hypot
import time
import os
from set_up_track import *

def subdivide():
    #sort boundaries so it is in order, cw from start
    #take 2 points, 20 appart, from each boundary
    #draw a box from these 4 points
    #make random colour for debug
    #create ordered list of boxes
    return (1)

def initial_track_line():
    #for each box, pick a random point
    # join points with a spline
    return (1)

def view_track_boundary(track_name, track_image):
    img_arr = pg.surfarray.pixels3d(track_image)
    try:
        track_order = np.load(f"orders/{track_name}_order.npy")
    except FileNotFoundError:
        print("FileNotFoundError: Track image not found.")
    for i in range(len(track_order)):
        temp = len(track_order[i])//24
        img_arr[i] = [temp, 255 - temp, 180]
    track_image = pygame.surfarray.make_surface(img_arr)
    return track_image

def main():
    start = time.time()
    # Initialise screen
    pg.init()
    screen = pg.display.set_mode((1920, 1080))
    pg.display.set_caption('Racing Lines')

    # Fill background
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill((250, 250, 250))


    track_name = 'monza' 
    order_of_operations = OrderOfOperations(track_name)
    order_of_operations.run()
    track_image = order_of_operations.get_track_image()
    track_image = view_track_boundary(track_name, track_image)



    print("Time taken to find order:", time.time() - start)



    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
        screen.blit(background, (0, 0))
        
        screen.blit(track_image, (500,300))
        
        pg.display.flip()


if __name__ == '__main__': main()