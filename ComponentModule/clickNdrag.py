import pygame as pg
#move feature
class Drag:
    def __init__(self):
        self.moving = False
        self.start = (0, 0)
        self.offset = (0, 0)

    def handle_event(self, event, rect, pos):

        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            print('clickyy')
            rect = rect.move(pos)
            if rect.collidepoint(event.pos):
                self.moving = True
                self.start = event.pos
                self.offset = pos

        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            print('clickyy up')
            self.moving = False

        elif event.type == pg.MOUSEMOTION and self.moving:
            print('dragyy')
            dx = event.pos[0] - self.start[0]
            dy = event.pos[1] - self.start[1]
            return (self.offset[0] + dx, self.offset[1] + dy)

        return pos


