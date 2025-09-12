import pygame as pg
from pygame.locals import *
import pygame


white, red, black, orange, green = (255, 255, 255), (255, 0, 0), (0, 0, 0), (255, 165, 0), (0, 255, 0)


def find_track_boundaries_with_background(image):
    #Find Vertical and horizontal side
    prev_black = 0 #boolean if the previous colour was black or not
    for i in range(2):
        order = [image.get_width(), image.get_height()]
        for y in range (order[0 + i]): #first does vertical side, then changes the order to do horizontal 
            for x in range (order[1 - i]):
                if pg.Surface.get_at(image, (x, y)) == black:
                    if prev_black == 0:
                        pg.set_at(image, (x, y), green) #change track boundary to green
                        prev_black = 1
                    else: #pixel colour is white or orange
                        prev_black = 0
                        if pg.Surface.get_at(image, (x, y)) == white:
                            pg.set_at(image, (x, y), orange) #change all white to orange cuz mclaren and show that its been checked

#def find_track_boundaries_no_bg(track):


def import_tracks():
    silverstone = pg.image.load("tracks/silverstone.png").convert_alpha()
    return silverstone

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
    silverstone = import_tracks()

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

