import pygame as pg
from pygame.locals import *
from ComponentModule.components import Text, EntryBox
from colours import colour_palette


# Using blit to copy content from one surface to other

def main():
    # Initialise screen
    pg.init()
    width = 1920
    height = 1080
    screen = pg.display.set_mode((width, height))
    pg.display.set_caption('Racing Lines')

    # Fill background
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill(colour_palette['BG_GREY'].value)

    # Display some text

    welcome = Text((width/2 - 100), (height//4), "Welcome", 72) 
    welcome.draw(background)

    logo = pg.image.load("Assets/F1logo.png").convert_alpha()
    input = EntryBox(700, 800, 500, 50)
    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == pg.MOUSEBUTTONDOWN:
                input.handle_event(event)
            if event.type == pg.KEYDOWN:
                input.handle_event(event)
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
        screen.blit(background, (0, 0))
        screen.blit(logo, (width/2 - logo.get_width()/2, height/2 - logo.get_height()/2))
        input.draw(screen)
        pg.display.flip()


if __name__ == '__main__': main()