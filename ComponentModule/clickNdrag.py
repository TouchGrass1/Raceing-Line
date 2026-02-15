import pygame as pg
#move feature
class Drag:
    def __init__(self):
        self.moving = False
        self.start_mouse = (0, 0)
        self.base_pan = (0, 0)

    def handle_event(self, event, clip_rect, current_pan):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if clip_rect.collidepoint(event.pos):
                self.moving = True
                self.start_mouse = event.pos
                self.base_pan = current_pan # Store where we started

        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.moving = False

        elif event.type == pg.MOUSEMOTION and self.moving:
            dx = event.pos[0] - self.start_mouse[0]
            dy = event.pos[1] - self.start_mouse[1]
            # Return the new cumulative pan
            return (self.base_pan[0] + dx, self.base_pan[1] + dy)

        return current_pan


