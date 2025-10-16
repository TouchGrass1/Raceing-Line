from __future__ import annotations


from typing import Optional, Tuple


import numpy as np
import pygame as pg


from .TrackLoader import TrackLoader
from .TrackProcessor import TrackProcessor
from .TrackValidator import TrackValidator




class OrderManager:
    def __init__(self, track_name: str, dark_mode: bool = False):
        self.track_name = track_name.lower()
        self.track_surface: Optional[pg.Surface] = None
        self.img_height: Optional[int] = None
        self.img_width: Optional[int] = None
        self.order: Optional[np.ndarray] = None
        self.dark_mode = dark_mode

    def run(self) -> None:
        result = TrackLoader.import_track(self.track_name)
        if result is None:
            print("Track import failed")
            return

        self.track_surface, self.img_height, self.img_width = result

        self.order = TrackLoader.load_order(self.track_name)
        if self.order is None:
            if TrackValidator.valid_track(self.track_surface):
                order = TrackProcessor.recolor_track_boundaries(self.track_surface, self.img_height, self.img_width)
                self.order = np.array(order, dtype=object)
                TrackLoader.save_order(self.order, self.track_name)
            else:
                print("Track validation failed - cannot compute order.")
                return

        TrackProcessor.toggle_background(self.track_surface, self.dark_mode)

    def get_track_surface(self) -> Optional[pg.Surface]:
        return self.track_surface

    def get_order(self) -> Optional[np.ndarray]:
        return self.order

    def get_dimensions(self) -> Optional[Tuple[int, int]]:
        if self.img_width is None or self.img_height is None:
            return None
        return self.img_width, self.img_height