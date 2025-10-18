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
        

        self.triangles = []
        self.spacing = 2 #spacing between points

    def resample(self, path):
        path = np.array(path, dtype=float)
        x, y = path[:, 0], path[:, 1]

        # Compute distances between consecutive points
        distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
        cumulative = np.concatenate(([0], np.cumsum(distances)))

        total_length = cumulative[-1]
        total_length = np.sum(np.sqrt(np.sum(np.diff(self.inner, axis=0)**2, axis=1))) #finds the difference between each point and sums them to get total length
        self.num_points = int(total_length / self.spacing)
        target_distances = np.linspace(0, total_length, self.num_points)

        # Interpolate x and y at new positions
        new_x = np.interp(target_distances, cumulative, x)
        new_y = np.interp(target_distances, cumulative, y)

        return np.stack((new_x, new_y), axis=1)
    
    def constructTriangles(self):
        for i in range(self.num_points -1):
            l1 = (self.inner_sample[i][0], self.inner_sample[i][1])
            l2 = (self.inner_sample[i + 1][0], self.inner_sample[i + 1][1])
            r1 = (self.outer_sample[i][0], self.outer_sample[i][1])
            r2 = (self.outer_sample[i + 1][0], self.outer_sample[i + 1][1])

            # Two triangles per quad
            self.triangles.append([l1, r1, r2])
            self.triangles.append([l1, r2, l2])

        return np.array(self.triangles, dtype=np.int32)

    def run(self):
        self.inner_sample = self.resample(self.inner)
        self.outer_sample = self.resample(self.outer)
        return self.constructTriangles()
    

    

def main():
    """Temporary visual test for the boundary viewer."""
    start = time.time()
    pg.init()
    screen = pg.display.set_mode((900, 800))
    pg.display.set_caption("Track Boundary Viewer")

    background = pg.Surface(screen.get_size()).convert()
    background.fill((20, 20, 20))

    track_name = "silverstone"
    viewer = ViewTrackBoundary(track_name)
    boundary_surface = viewer.view_track_boundary()
    triangles = Subdivide(track_name).run()
    

    print("Boundary generated in:", round(time.time() - start, 2), "seconds")

    running = True
    while running:
        for event in pg.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

        screen.blit(background, (0, 0))
        screen.blit(boundary_surface, (0, 0))
        for i in range(1, len(triangles) - 1):
            pg.draw.polygon(screen, (255,0,0), triangles[i], 2)
        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    main()
