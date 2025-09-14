import pygame as pg
from pygame.locals import *
import numpy as np
from math import hypot
import time

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

        flood_fill(img_arr, start_blue, start_green)


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
    find_order(img_arr, start_blue, start_green, colours)

          
def find_order(img_arr, start_blue, start_green, colours):
    start = time.time()
    order = [[start_blue], [start_green]]

    for i in range(2):
        run = True
        while run == True:

            node = order[i][-1]
            neighbors = candidate_neighbors(node, size=5)
            closest = None
            closest_dist = np.inf
            blues_left = False #to check if there are any blues left
            for neighbor in neighbors:
                if 0 <= neighbor[0] < img_arr.shape[0] and 0 <= neighbor[1] < img_arr.shape[1]:
                    if np.array_equal(img_arr[neighbor[0], neighbor[1]], colours[i]):
                        blues_left = True
                        dist = hypot(node[0] - neighbor[0], node[1] - neighbor[1])
                        if 0 < dist < closest_dist: #not the same point
                            closest = (neighbor[0], neighbor[1])
                            closest_dist = dist
                            if closest_dist == 1:
                                break #possible optimization
            if blues_left == False:
                node = order[i][-1] #double check with larger search area
                neighbors = candidate_neighbors(node, size=11)
                closest = None
                closest_dist = np.inf
                blues_left = False #to check if there are any blues left
                for neighbor in neighbors:
                    if 0 <= neighbor[0] < img_arr.shape[0] and 0 <= neighbor[1] < img_arr.shape[1]:
                        if np.array_equal(img_arr[neighbor[0], neighbor[1]], colours[i]):
                            blues_left = True
                            dist = hypot(node[0] - neighbor[0], node[1] - neighbor[1])
                            if 0 < dist < closest_dist: #not the same point
                                closest = (neighbor[0], neighbor[1])
                                closest_dist = dist
                                if closest_dist == 1:
                                    break #possible optimization
                if blues_left == False:
                    run = False #final exit
            if closest is not None:
                order[i].append(closest)
                temp = len(order[i])//24
                img_arr[closest] = [temp, 255 - temp, 180] #change colour
    print("Time taken to find order:",time.time() - start)
    return order





def import_tracks():
    silverstone = pg.image.load("tracks/silverstone.png").convert_alpha()
    height = silverstone.get_height()
    width = silverstone.get_width()
    return silverstone, height, width

class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pg.Rect(x, y, w, h)
        self.checked = False
        self.label = label
    def draw(self, screen, font):
        pg.draw.rect(screen, (88, 101, 242), self.rect, width = 2)
        if self.checked:
            pg.draw.rect(screen, (255, 255, 255), self.rect)
        label_text = font.render(self.label, True, (0,0,0))
        label_rect = label_text.get_rect(center=self.rect.center)
        screen.blit(label_text, (self.rect.x + 3, self.rect.y + 3))




def main():
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
    silverstone, height, width = import_tracks()
    colour_track_boundaries(silverstone, height, width)



    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
        screen.blit(background, (0, 0))
        screen.blit(silverstone, (500,300))

        #iport_button.draw(screen, font)

        
        pg.display.flip()


if __name__ == '__main__': main()

