from set_up_track import import_tracks
import pygame as pg
from pygame.locals import *
from colours import colour_pallete
import time
from ComponentModule.zoom import *
from ComponentModule.clickNdrag import *

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

    zoom = Zoom(track_image)
    drag = Drag()
    pos = (500, 400)
    change = (0, 0)
    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            elif event.type == MOUSEWHEEL:
                zoom.handle_event(event)
 



        # Draw
        img = zoom.get_image()
        rect = img.get_rect()
        pos = drag.handle_event(event, rect, pos)
        screen.blit(background, (0, 0))
        screen.blit(img, pos)
        pg.display.flip()
        
        pg.display.flip()


if __name__ == '__main__': main()