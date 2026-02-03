from __future__ import annotations


import os
from pathlib import Path
from typing import Optional, Tuple


import numpy as np
import pygame as pg


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = _PROJECT_ROOT / "assets" / "tracks"
ORDERS_DIR = _PROJECT_ROOT / "orders"
ORDERS_DIR.mkdir(exist_ok=True)




class TrackLoader:
    @staticmethod
    def import_track(track_name = "qatar"):
        """Load a track image from `assets/tracks/{track_name}.png`.


        Returns (surface, height, width) or None if not found.
        """
        filepath = ASSETS_DIR / f"{track_name}.png"
        try:
            surface = pg.image.load(str(filepath)).convert_alpha()
        except FileNotFoundError:
            print(f"Track file not found: {filepath}")
            return None
        return surface, surface.get_height(), surface.get_width()


    @staticmethod
    def save_order(order: object, track_name: str) -> None:
        filename = ORDERS_DIR / f"{track_name}_order.npy"
        np.save(filename, order, allow_pickle=True)
        print(f"Order saved to {filename}")


    @staticmethod
    def load_order(track_name: str) -> Optional[object]:
        filename = ORDERS_DIR / f"{track_name}_order.npy"
        try:
            return np.load(filename, allow_pickle=True)
        except FileNotFoundError:
            print(f"No saved order found: {filename}")
            return None