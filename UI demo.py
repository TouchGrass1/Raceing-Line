import pygame as pg
from pygame.locals import *
import numpy as np

colour_pallete = {
    'white': (242, 241, 242),
    'red': (246, 32, 57),
    'black': (17, 17, 17),
    'blue': (41, 82, 148),
    'BG grey': (2, 29, 33),
    'orange': (255, 128, 0),
    'red2': (187, 57, 59),
    'line grey': (42, 45, 49),
    'subtle grey': (37, 40, 44)
}

class Button:
    def __init__(self, x, y, w, h, label):
        self.rect = pg.Rect(x, y, w, h)
        self.checked = False
        self.label = label
    def draw(self, screen, font):
        pg.draw.rect(screen, (88, 101, 242), self.rect, width = 2)
        if self.checked:
            pg.draw.rect(screen, (255, 255, 255), self.rect)
        label_text = font.render(self.label, True, (0,0,0))
        label_rect = label_text.get_rect(center=self.rect.center)
        screen.blit(label_text, (self.rect.x + 3, self.rect.y + 3))

class divider:
    def __init__(self, x, y, w, h):
        self.rect = pg.Rect(x, y, w, h)
    def draw(self, screen):
        pg.draw.rect(screen, colour_pallete['line grey'], self.rect)
        
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label, active):
        self.active = active
        self.rect = pg.Rect(x, y, w, h)
        self.min = min_val
        self.max = max_val
        self.val = initial_val
        self.oldval = initial_val
        self.label = label
        self.dragging = False
        self.handle_x = x + (initial_val - min_val)/(max_val - min_val) * w
        self.font = pg.font.Font(None, 30)

    def draw(self, screen, font):
        # Draw slider track
        pg.draw.rect(screen, colour_pallete['subtle grey'], self.rect, border_radius=23)

        
        if self.active:        
            # Draw handle
            pg.draw.circle(screen, (255, 255, 255), 
                            (int(self.handle_x), self.rect.centery), 10)
            
            # Draw labels
            label_text = font.render(self.label, True, (255,255,255))
            screen.blit(label_text, (self.rect.x - 20, self.rect.y - 30))
            
            min_text = font.render(f"{self.min:.1f}", True, (255,255,255))
            screen.blit(min_text, (self.rect.left - 40, self.rect.centery))
            
            max_text = font.render(f"{self.max:.1f}", True, (255,255,255))
            screen.blit(max_text, (self.rect.right + 10, self.rect.centery ))
            
            val_text = font.render(f"{self.val:.2f}", True, (255,255,255))
            screen.blit(val_text, (self.handle_x - 15, self.rect.bottom + 5))
        
        else:
            x = (self.val - self.min)/(self.max - self.min) * self.rect.width
            rect = pg.Rect(self.rect.left, self.rect.top, x, self.rect.height )
            pg.draw.rect(screen, colour_pallete['red2'], rect, border_radius=23)
            
            val_text = self.font.render(f"{self.val*100:.2n}%", 1, colour_pallete['white'])
            pos = (self.rect.right - val_text.get_width(), self.rect.bottom + val_text.get_height())
            screen.blit(val_text, pos)


    def update_value(self, mouse_x):
        if self.active:
            self.handle_x = np.clip(mouse_x, self.rect.left, self.rect.right)
            self.val = self.min + (self.handle_x - self.rect.left)/self.rect.width * (self.max - self.min)
        else:
            pass

class Text:
    def __init__(self, x, y, text, font_size):
        self.pos = (x, y)
        self.text = text
        self.font = pg.font.Font(None, font_size)
    def draw(self, background):
        text = self.font.render(str(self.text), 1, colour_pallete['white'])
        background.blit(text, self.pos)
        
        
def main():
    # Initialise screen
    pg.init()
    screen = pg.display.set_mode((1920, 1080))
    pg.display.set_caption('Racing Lines')

    # Fill background
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill(colour_pallete['BG grey'])

    # Display some text
    font = pg.font.Font(None, 36)
    throttle_text = Text(1547, 182, "Throttle", 36) 
    throttle_text.draw(background)

    

    iport_button = Button(100, 200, 200, 50, "button")
    divididers = [divider(0, 180, 1538, 3), divider(1535, 0, 3, 1080), divider(200, 180, 3, 900), divider(1538, 750, 383, 3)]   
    throttle_slider = Slider(1547, 223, 360, 32, 0, 1, 0.8, "Throttle", False)


    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
        screen.blit(background, (0, 0))
        for d in divididers:
            d.draw(screen)
            

        iport_button.draw(screen, font)
        throttle_slider.draw(screen, font)
        
        pg.display.flip()


if __name__ == '__main__': main()
