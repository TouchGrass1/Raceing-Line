import pygame as pg
from pygame.locals import *
import numpy as np
from math import hypot
import time
import os

white  = np.array([255, 255, 255], dtype=np.uint8)
red    = np.array([255, 0, 0], dtype=np.uint8)
black  = np.array([0, 0, 0], dtype=np.uint8)
orange = np.array([255, 165, 0], dtype=np.uint8)
green  = np.array([0, 180, 75], dtype=np.uint8)
blue   = np.array([0, 81, 186], dtype=np.uint8)
blue2 = (0, 81, 186)
green2 = (0, 180, 75)


def candidate_neighbors(node, size = 3):
    nodes = []
    for i in range(size):
        for j in range(size):
            nodes.append((node[0]+i-int(size/2), node[1]+j-int(size/2)))
    return (nodes)

def join_gap(node1, node2):
    pg.draw.line(track_image, blue, node1, node2)

def colour_track_boundaries(image, height, width):
        img_arr = pg.surfarray.pixels3d(image) #using surfarray to access pixel values, more efficient as written in C
        
        #create masks
        mask_white  = np.all(img_arr == white, axis=2)
        mask_black  = np.all(img_arr == black, axis=2)
        mask_blue   = np.all(img_arr == blue, axis=2)
        mask_green  = np.all(img_arr == green, axis=2)
        
        #apply masks
        img_arr[mask_black] = white
        img_arr[~(mask_white | mask_black | mask_blue | mask_green)] = orange

        for y in range (height): #first does vertical side, then changes the order to do horizontal 
            for x in range (width):
                if pg.Surface.get_at(image, (x, y)) == blue2: # 2 dots of blue to signify start
                    start_blue = [x , y]
                    continue
                elif pg.Surface.get_at(image, (x, y)) == green2:
                    start_green = [x , y]
                    continue

        order = flood_fill(img_arr, start_blue, start_green)
        return order


#NOTE: all colouring is done just as a visual aid for debugging, will be removed to imporove performance

def flood_fill(img_arr, start_blue, start_green):


    sets = [set(), set()]
    sets[0].add((start_blue[0], start_blue[1]))
    sets[1].add((start_green[0], start_green[1]))
    colours = [blue, green]

    for i, my_set in enumerate(sets):
        while my_set:
            node = my_set.pop()
            img_arr[node] = colours[i]

            for nx, ny in candidate_neighbors(node, size=5):
                if 0 <= nx < img_arr.shape[0] and 0 <= ny < img_arr.shape[1]:
                    if np.array_equal(img_arr[nx, ny], orange):
                        my_set.add((int(nx), int(ny))) 

    # Ensure tuple conversion when passing
    order = find_order(img_arr, start_blue, start_green, colours)
    return order

          
def find_order(img_arr, start_blue, start_green, colours):
    
    order = [[start_blue], [start_green]]

    for i in range(2):
        run = True
        while run:
            node = order[i][-1]
            closest = None
            closest_dist = np.inf

            for size in [3, 5, 11]: # Loop through progressively larger neighborhood sizes
                neighbors = candidate_neighbors(node, size=size)
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < img_arr.shape[0] and 0 <= neighbor[1] < img_arr.shape[1]:
                        if np.array_equal(img_arr[neighbor[0], neighbor[1]], colours[i]):
                            dist = hypot(node[0] - neighbor[0], node[1] - neighbor[1])
                            if 0 < dist < closest_dist:
                                closest = (neighbor[0], neighbor[1])
                                closest_dist = dist
                                if closest_dist == 1:
                                    break
                if closest is not None:
                    if size == 11:
                        join_gap(order[i][-1], closest)
                    break  # stop checking larger sizes

            # If no match was found in any size, stop the run
            if closest is None:
                run = False
            else:
                order[i].append(closest)
                temp = len(order[i]) // 24
                img_arr[closest] = [temp, 255 - temp, 180]  # update colour

    
    check_return_to_start(order, start_blue, start_green)
    return order

#validation
def valid_track(image):
    img_arr = pg.surfarray.pixels3d(image)
    

    match_blue = np.all(img_arr == blue, axis = -1)
    match_green = np.all(img_arr == blue, axis = -1)
    blue_count = np.sum(match_blue)
    green_count = np.sum(match_green)

    if blue_count == 1 and green_count == 1:
        return True

    else: 
        print('you have',blue_count,'blues and',green_count,'greens \n blue value: ',blue, 'green value: ',green)
        return False

def check_return_to_start(order, start_blue, start_green):
    start = [start_blue, start_green]
    for i in range(2):
        if hypot(order[i][-1][0] - start[i][0], order[i][-1][1] - start[i][1]) < 1.5:
            print("Returned to start")
            return True

#load and save   
def save_order(order, track_name):
    filename = f"{track_name}_order.npy"
    np.save(os.path.join('orders', filename), order)
    print(f"Order saved as {filename}")

def load_order(track_name):
    filename = f"{track_name}_order.npy"
    try:
        return np.load(os.path.join('orders', filename), allow_pickle=True)
    except FileNotFoundError:
        print("No saved order found")
        return None
    
def import_tracks(track_name= 'silverstone'):
    global track_image
    track_image  = pg.image.load(f"Assets/tracks/{track_name}.png").convert_alpha()
    height = track_image.get_height()
    width = track_image.get_width()
    return track_image, height, width






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

    # Display some text
    font = pg.font.Font(None, 36)
    text = font.render("Hello There", 1, (10, 10, 10))
    textpos = text.get_rect()
    textpos.centerx = background.get_rect().centerx
    background.blit(text, textpos)

    #iport_button = Button(100, 200, 200, 50, "button")
    track_name = 'silverstone' 
    track_image, height, width = import_tracks(track_name)
    order = load_order(track_name)
    if order is None:
        if valid_track(track_image):
            order = colour_track_boundaries(track_image, height, width)
            order = np.array(order, dtype=object)
            save_order(order, track_name)

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

