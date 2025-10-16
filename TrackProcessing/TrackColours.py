from __future__ import annotations


import numpy as np

# Colours as numpy arrays (dtype uint8) for surfarray pixel comparisons
WHITE = np.array([255, 255, 255], dtype=np.uint8)
RED = np.array([255, 0, 0], dtype=np.uint8)
BLACK = np.array([0, 0, 0], dtype=np.uint8)
ORANGE = np.array([255, 165, 0], dtype=np.uint8)
GREEN = np.array([0, 180, 75], dtype=np.uint8)
BLUE = np.array([0, 81, 186], dtype=np.uint8)


# Pygame-friendly RGB tuples for get_at / draw operations
BLUE_RGB = (0, 81, 186)
GREEN_RGB = (0, 180, 75)
BG_GREY = (2, 29, 33)