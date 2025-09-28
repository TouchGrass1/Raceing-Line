import pygame as pg
#move feature
class Drag:
    def __init__(self):
        self.moving = False
    def handle_event(self, event, rect):
        if event.type == pg.MOUSEBUTTONDOWN:
            if rect.collidepoint(event.pos):
                moving = True
        elif event.type == pg.MOUSEBUTTONUP:
            moving = False

        # Make your image move continuously
        elif event.type == pg.MOUSEMOTION and moving:
            rect.move_ip(event.rel)



