import pygame as pg
from pygame.locals import *
from ComponentModule.components import Text, EntryBox
from colours import colour_palette


# Using blit to copy content from one surface to other

def main():
    # Initialise screen
    pg.init()
    width = 1280
    height = 720
    screen = pg.display.set_mode(flags=pg.FULLSCREEN)
    #screen = pg.display.set_mode((width, height))
    screen_shape = screen.get_size()
    width, height = screen_shape
    pg.display.set_caption('Racing Lines')
    font_size = height//30
    font = pg.font.Font(None, font_size)

    # Fill background
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill(colour_palette['BG_GREY'].value)

    # Display some text
    welcome = Text((width/2 - 100), (height//4), "Welcome", 72) 
    welcome.draw(background)

    logo = pg.image.load("Assets/F1logo.png").convert_alpha()
    input = EntryBox((0.3645 * width), (0.74*height), (0.26*width), (0.046296*height), font, is_password=True)
    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == pg.MOUSEBUTTONDOWN:
                input.handle_event(event)
            if event.type == pg.KEYDOWN:
                input.handle_event(event)
            
            #check for quits
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            
        #draw background and logo
        screen.blit(background, (0, 0))
        screen.blit(logo, (width/2 - logo.get_width()/2, height/2 - logo.get_height()/2))
        input.draw(screen)
        if input.is_correct:
            return True
        #update
        pg.display.flip()


#if __name__ == '__main__': main()