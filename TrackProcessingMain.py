from __future__ import annotations

import time
import pygame as pg
from pygame.locals import QUIT, KEYDOWN, K_ESCAPE

from TrackProcessing.OrderManager import OrderManager


def main():
    start_time = time.time()
    pg.init()

    screen = pg.display.set_mode((1280, 720))
    pg.display.set_caption("Racing Lines - Refactor")

    background = pg.Surface(screen.get_size()).convert()
    background.fill((250, 250, 250))

    track_name = "monza"
    manager = OrderManager(track_name)
    manager.run()

    track_surface = manager.get_track_surface()

    print("Time taken:", time.time() - start_time)

    running = True
    while running:
        for event in pg.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        screen.blit(background, (0, 0))
        if track_surface is not None:
            screen.blit(track_surface, (100, 100))
        pg.display.flip()

    pg.quit()


if __name__ == "__main__":
    main()


