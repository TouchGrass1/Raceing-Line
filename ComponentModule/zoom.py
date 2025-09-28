import cv2
import pygame as pg

#Zoom feature
class Zoom:
# cv2.INTER_AREA (best for shrinking) and cv2.INTER_CUBIC (best for enlarging)
    def __init__(self, image):
        self.image = image
        self.x,self.y = image.shape[1], image.shape[0]
        self.scale = self.x/self.y
        self.speed = 1
    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up only counts one scroll, as it counts as one event
                        new_size = (self.x+self.scale*self.speed, self.y+self.speed)
                        self.resized_image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_CUBIC)
                    elif event.button == 5:  # Scroll down
                        new_size = (self.x-self.scale*self.speed, self.y-self.speed)
                        self.resized_image = cv2.resize(self.image, new_size, interpolation=cv2.INTER_AREA)
        return self.resized_image