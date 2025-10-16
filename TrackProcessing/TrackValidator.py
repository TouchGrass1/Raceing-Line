from __future__ import annotations


from math import hypot
from typing import List, Tuple


import numpy as np
import pygame as pg


from .TrackColours import BLUE, GREEN




class TrackValidator:
    @staticmethod
    def valid_track(surface: pg.Surface) -> bool:
        """Check that the surface contains exactly one blue and one green start pixel.


        This function uses surfarray to count exact matches of the colour arrays.
        """
        img_arr = pg.surfarray.pixels3d(surface)


        match_blue = np.all(img_arr == BLUE, axis=-1)
        match_green = np.all(img_arr == GREEN, axis=-1)


        blue_count = int(np.sum(match_blue))
        green_count = int(np.sum(match_green))


        if blue_count == 1 and green_count == 1:
            return True


        print(
        "Invalid start colours detected:",
        f"blue_count={blue_count}, green_count={green_count}",
        )
        return False


    @staticmethod
    def check_return_to_start(order: List[List[Tuple[int, int]]],
        start_blue: Tuple[int, int],
        start_green: Tuple[int, int]) -> bool:
        """Validate that each path returns to its start (within 1.5 pixels)."""
        start_points = [start_blue, start_green]
        for i in range(2):
            if hypot(order[i][-1][0] - start_points[i][0], order[i][-1][1] - start_points[i][1]) < 1.5:
                print("Returned to start for path", i)
                return True
        return False