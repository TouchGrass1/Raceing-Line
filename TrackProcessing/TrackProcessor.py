
from __future__ import annotations

from math import hypot
from typing import List, Optional, Tuple

import numpy as np
import pygame as pg

from .TrackColours import WHITE, BLACK, BLUE, GREEN, ORANGE, BG_GREY, BLUE_RGB, GREEN_RGB
from .TrackValidator import TrackValidator


class TrackProcessor:
    @staticmethod
    def get_candidate_neighbors(node: Tuple[int, int], size: int = 3) -> List[Tuple[int, int]]:
        """Return coordinates surrounding `node` using a `size x size` square."""
        cx, cy = node
        offset = size // 2
        return [(cx + dx - offset, cy + dy - offset) for dx in range(size) for dy in range(size)]

    @staticmethod
    def join_gap(node1: Tuple[int, int], node2: Tuple[int, int], surface: pg.Surface) -> None:
        pg.draw.line(surface, BLUE_RGB, node1, node2)

    @staticmethod
    def recolor_track_boundaries(surface: pg.Surface, img_height: int, img_width: int):
        """Colour the track boundaries and produce an ordered path for both directions.

        Returns `order` as a list containing two lists of (x,y) tuples.
        """
        img_arr = pg.surfarray.pixels3d(surface)

        mask_white = np.all(img_arr == WHITE, axis=2)
        mask_black = np.all(img_arr == BLACK, axis=2)
        mask_blue = np.all(img_arr == BLUE, axis=2)
        mask_green = np.all(img_arr == GREEN, axis=2)

        img_arr[mask_black] = WHITE
        img_arr[~(mask_white | mask_black | mask_blue | mask_green)] = ORANGE

        start_blue = start_green = None
        for y in range(img_height):
            for x in range(img_width):
                pixel = surface.get_at((x, y))[:3]
                if pixel == BLUE_RGB:
                    start_blue = (x, y)
                elif pixel == GREEN_RGB:
                    start_green = (x, y)

        if start_blue is None or start_green is None:
            raise ValueError("Missing start markers (blue/green) on the track surface")

        order = TrackProcessor._flood_fill(img_arr, start_blue, start_green)
        return order

    @staticmethod
    def _flood_fill(img_arr: np.ndarray, start_blue: Tuple[int, int], start_green: Tuple[int, int]):
        sets = [set(), set()]
        sets[0].add((start_blue[0], start_blue[1]))
        sets[1].add((start_green[0], start_green[1]))
        colours = [BLUE, GREEN]

        for i, pixel_set in enumerate(sets):
            while pixel_set:
                node = pixel_set.pop()
                img_arr[node] = colours[i]

                for nx, ny in TrackProcessor.get_candidate_neighbors(node, size=5):
                    if 0 <= nx < img_arr.shape[0] and 0 <= ny < img_arr.shape[1]:
                        if np.array_equal(img_arr[nx, ny], ORANGE):
                            pixel_set.add((int(nx), int(ny)))

        order = TrackProcessor._find_order(img_arr, start_blue, start_green, colours)
        return order

    @staticmethod
    def _find_order(img_arr: np.ndarray, start_blue: Tuple[int, int], start_green: Tuple[int, int], colours: List[np.ndarray]):
        order: List[List[Tuple[int, int]]] = [[start_blue], [start_green]]

        for i in range(2):
            run = True
            while run:
                node = order[i][-1]
                closest: Optional[Tuple[int, int]] = None
                closest_dist = float("inf")

                for size in [3, 5, 11]:
                    neighbors = TrackProcessor.get_candidate_neighbors(node, size=size)
                    for neighbor in neighbors:
                        if (0 <= neighbor[0] < img_arr.shape[0] and
                                0 <= neighbor[1] < img_arr.shape[1]):
                            if np.array_equal(img_arr[neighbor[0], neighbor[1]], colours[i]):
                                dist = hypot(node[0] - neighbor[0], node[1] - neighbor[1])
                                if 0 < dist < closest_dist:
                                    closest = (neighbor[0], neighbor[1])
                                    closest_dist = dist
                                    if closest_dist == 1:
                                        break
                    if closest is not None:
                        if size == 11:
                            TrackProcessor.join_gap(order[i][-1], closest, pg.display.get_surface() or pg.Surface((1,1)))
                        break

                if closest is None:
                    run = False
                else:
                    order[i].append(closest)
                    temp = len(order[i]) // 24
                    img_arr[closest] = [temp, 255 - temp, 180]

        TrackValidator.check_return_to_start(order, start_blue, start_green)
        return order

    @staticmethod
    def toggle_background(surface: pg.Surface, dark_mode: bool) -> None:
        img_arr = pg.surfarray.pixels3d(surface)
        if dark_mode:
            mask = np.all(img_arr == WHITE, axis=2)
            img_arr[mask] = BG_GREY
        else:
            mask = np.all(img_arr == BG_GREY, axis=2)
            img_arr[mask] = WHITE
