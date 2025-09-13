import pygame as pg
from pygame.locals import *



white, red, black, orange, green, clear, blue = (255, 255, 255), (255, 0, 0), (0, 0, 0), (255, 165, 0), (0, 180, 75), (0, 0, 0, 0), (0, 120, 215)


def candidate_neighbors(node):
    return ((node[0]-1, node[1]-1), (node[0]-1, node[1]), (node[0]-1, node[1]+1), (node[0], node[1]-1), 
            (node[0], node[1]+1), (node[0]+1, node[1]-1), (node[0]+1, node[1]), (node[0]+1, node[1]+1))

def colour_track_boundaries(image, height, width):
        for y in range (height): #first does vertical side, then changes the order to do horizontal 
            for x in range (width):
                if pg.Surface.get_at(image, (x, y))[3] == 0:
                     continue
                elif pg.Surface.get_at(image, (x, y)) == blue or pg.Surface.get_at(image, (x, y)) == green: # 2 dots of blue to signify start
                     start = [x , y]
                     continue
                elif pg.Surface.get_at(image, (x, y)) == black:
                        pg.Surface.set_at(image, (x, y), clear)
                else:
                    pg.Surface.set_at(image, (x, y), red) #change all rest to orange cuz mclaren and show that its been checked
        differentiate_track_boundaries(image, height, width)

def differentiate_track_boundaries(image, height, width):
    outer_boundary = set()
    inner_boundary = set()
    for y in range (height): # the error with this method is that it only passes each pixel once, if its neighbor changes colour, but it has already been passed, it wont be changed, so should start with blue/ green dots and then work outwards using new tenchnique called flood fille
        for x in range (width):
            if pg.Surface.get_at(image, (x, y))[3] == orange:
                for neighbor in candidate_neighbors((x, y)):
                    if pg.Surface.get_at(image, neighbor) == blue:
                        pg.Surface.set_at(image, (x, y), blue)
                        outer_boundary.add((x, y))
                    elif pg.Surface.get_at(image, neighbor) == green:
                        pg.Surface.set_at(image, (x, y), green)
                        inner_boundary.add((x, y))




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

