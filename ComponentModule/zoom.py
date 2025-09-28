import cv2
import pygame as pg
from pygame.locals import *
import numpy as np

def surface_to_cvimage(surface):
    data = pg.image.tostring(surface, "RGB")
    img = np.frombuffer(data, dtype=np.uint8)
    img = img.reshape((surface.get_height(), surface.get_width(), 3))
    return img

def cvimage_to_surface(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    surf = pg.image.frombuffer(img.tobytes(), img.shape[1::-1], "RGB")
    return surf


class Zoom:
    def __init__(self, image):
        # convert pygame surface to cv2 image
        self.cv_image = surface_to_cvimage(image)
        self.h, self.w = self.cv_image.shape[:2]
        self.ratio = self.w / self.h
        self.scale = 1.0
        self.speed = 0.1
        self.resized_image = image
    


    def handle_event(self, event):
        if event.type == MOUSEWHEEL:
            if event.y > 0:  # Scroll up
                self.scale += self.speed
            else:  # Scroll down
                self.scale = max(0.1, self.scale - self.speed)

            new_w = int(self.w * self.scale)
            new_h = int(self.h * self.scale)
            resized_cv = cv2.resize(self.cv_image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            self.resized_image = cvimage_to_surface(resized_cv)

        return self.resized_image

    def get_image(self):
        return self.resized_image
