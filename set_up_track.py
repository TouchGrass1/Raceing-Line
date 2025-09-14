import pygame as pg
from pygame.locals import *
from numpy import hypot, sqrt


white, red, black, orange, green, clear, blue = (255, 255, 255), (255, 0, 0), (0, 0, 0), (255, 165, 0), (0, 180, 75), (0, 0, 0, 0), (0, 81, 186)


def candidate_neighbors(node, size = 3):
    nodes = []
    for i in range(size):
        for j in range(size):
            nodes.append((node[0]+i-int(size/2), node[1]+j-int(size/2)))
    return (nodes)

def colour_track_boundaries(image, height, width):
        for y in range (height): #first does vertical side, then changes the order to do horizontal 
            for x in range (width):
                if pg.Surface.get_at(image, (x, y)) == white:
                    pg.Surface.set_at(image, (x, y), clear)
                elif pg.Surface.get_at(image, (x, y)) == black:
                    continue
                elif pg.Surface.get_at(image, (x, y)) == blue : # 2 dots of blue to signify start
                    start_blue = [x , y]
                    continue
                elif pg.Surface.get_at(image, (x, y)) == green:
                    start_green = [x , y]
                    continue
                else:
                    pg.Surface.set_at(image, (x, y), orange) #change all rest to orange cuz mclaren and show that its been checked
        flood_fill(image, start_blue, start_green)

#NOTE: all colouring is done just as a visual aid for debugging, will be removed to imporove performance

def flood_fill(image, start_blue, start_green):
    sets = [set(), set()]

    sets[0].add((start_blue[0], start_blue[1]))
    sets[1].add((start_green[0], start_green[1]))
    colour = [blue, green]
    for i, my_set in enumerate(sets):
        while len(my_set) > 0:
            node = my_set.pop()
            pg.Surface.set_at(image, (node[0], node[1]), colour[i])
            neighbors = candidate_neighbors(node, size= 5)
            for neighbor in neighbors:
                if neighbor[0] < 0 or neighbor[1] < 0 or neighbor[0] >= image.get_width() or neighbor[1] >= image.get_height():
                    continue
                elif pg.Surface.get_at(image, (neighbor[0], neighbor[1])) == orange:
                    my_set.add((neighbor[0], neighbor[1]))
          

  




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

