import time
import pygame as pg
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE
from TrackProcessing.OrderManager import OrderManager

###NOTE each class will be a seperate file but the Trackproccessing import doesnt work if it is in a sibling module do need to fix but not important
class ViewTrackBoundary:
    def __init__(self, track_name: str):
        self.manager = OrderManager(track_name)
        self.manager.run()
        self.track_order = self.manager.get_order()
        self.track_name = track_name

        # Use track dimensions even though we won't display the texture
        dims = self.manager.get_dimensions()
        if dims is None:
            raise ValueError("Failed to get track dimensions.")
        self.width, self.height = dims

    def view_track_boundary(self) -> pg.Surface:

        if self.track_order is None:
            raise ValueError("Track order not loaded correctly.")

        print("Order lengths:", [len(p) for p in self.track_order])
        print("First 5 points:", self.track_order[0][:5])



        boundary_surface = pg.Surface((self.width, self.height), pg.SRCALPHA)

        # Draw boundaries
        for path in self.track_order:
            for i in range(1, len(path)):
                temp = len(self.track_order[0][i])//24
                boundary_color = (temp, 255 - temp, 180)
                pg.draw.line(boundary_surface, boundary_color, path[i - 1], path[i], width=2)

        return boundary_surface

import numpy as np
class Subdivide:
    def __init__(self, track_name: str):
        self.manager = OrderManager(track_name)
        self.manager.run()
        self.track_order = self.manager.get_order()
        self.inner, self.outer = self.track_order
        self.n = min(len(self.inner), len(self.outer)) - 1

        self.triangles = []
        self.width = 20 #triangle width

    def triangles(self):
        for i in range(self.n):
            l1 = (self.inner[i][0], self.inner[i][1])
            l2 = (self.inner[i + self.width][0], self.inner[i + self.width][1])
            r1 = (self.outer[i][0], self.outer[i][0])
            r2 = (self.outer[i + self.width][0], self.outer[i + self.width][1])

            # Two triangles per quad
            self.triangles.append([l1, r1, r2])
            self.triangles.append([l1, r2, l2])

        return np.array(self.triangles, dtype=np.int32)
    

def main():
    """Temporary visual test for the boundary viewer."""
    start = time.time()
    pg.init()
    screen = pg.display.set_mode((1920, 1080))
    pg.display.set_caption("Track Boundary Viewer")

    background = pg.Surface(screen.get_size()).convert()
    background.fill((20, 20, 20))

    track_name = "silverstone"
    viewer = ViewTrackBoundary(track_name)
    boundary_surface = viewer.view_track_boundary()
    

    print("Boundary generated in:", round(time.time() - start, 2), "seconds")

    running = True
    while running:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        screen.blit(background, (0, 0))
        screen.blit(boundary_surface, (500, 300))
        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    main()
