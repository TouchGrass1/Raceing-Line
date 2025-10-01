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
        self.clicked = False
        self.clicking = False # To handle holding down the button
        self.hover = False
        self.label = label
    def draw(self, screen, font):
        if self.clicking: colour = colour_pallete['blue']
        elif self.hover: colour = colour_pallete['red2']
        else: colour = colour_pallete['red']
        pg.draw.rect(screen, colour, self.rect, border_radius= 20)
        label_text = font.render(self.label, True, colour_pallete['white'])
        label_rect = label_text.get_rect(center=self.rect.center)
        screen.blit(label_text, label_rect)
        
    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicking = True
        if event.type == pg.MOUSEBUTTONUP:
            if self.clicking and self.rect.collidepoint(event.pos):
                self.clicked = not self.clicked
            self.clicking = False
        
        if event.type == pg.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self.hover = True
            else:
                self.hover = False

class Dropdown(Button):
    def __init__(self, x, y, w, h, label, options):
        super().__init__(x, y, w, h, label)
        self.options = options
        self.expanded = False
        self.selected_option = label
        print('1', self.selected_option)
        self.hover_options = -1
        self.option_rects = [pg.Rect(x, y + (i+1)*h, w, h) for i in range(len(options))]
    def draw(self, screen, font):
        
        if self.expanded:
            pg.draw.rect(screen, colour_pallete['blue'], self.rect, border_top_left_radius= 20, border_top_right_radius= 20)
            label_text = font.render(self.label, True, colour_pallete['white'])
            label_rect = label_text.get_rect(center=self.rect.center)
            screen.blit(label_text, label_rect)
            for i, option in enumerate(self.options):
                if i == self.hover_options: colour = colour_pallete['red2'] #hover
                else: colour = colour_pallete['line grey']
                if i == len(self.options) - 1:
                    pg.draw.rect(screen, colour, self.option_rects[i], border_bottom_left_radius= 20, border_bottom_right_radius= 20) #last option
                else:
                    pg.draw.rect(screen, colour, self.option_rects[i])
                option_text = font.render(option, True, colour_pallete['white'])
                option_rect = option_text.get_rect(center=self.option_rects[i].center)
                screen.blit(option_text, option_rect)
        else: super().draw(screen, font)
                
    def handle_event(self, event):
        
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos) and not self.expanded: #click once and hold on main button:
                self.expanded = True

            elif self.expanded: #click on options
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos) and self.options[i] != self.selected_option:
                        self.selected_option = self.options[i]
                        self.label = self.selected_option
                        self.expanded = False
                        print('2', self.selected_option)
        if event.type == pg.MOUSEBUTTONUP and not self.rect.collidepoint(event.pos):
                self.expanded = False

        if event.type == pg.MOUSEMOTION and self.expanded:
            self.hover_options = -1
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(event.pos):
                    self.hover_options = i
                    break
                


    def get_track(self):
        return self.selected_option


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

class EntryBox:
    def __init__(self, x, y, w, h, text='', pwd='asdf', placeholder='Type asdf to start'):
        self.rect = pg.Rect(x, y, w, h)
        self.color_inactive = colour_pallete['line grey']
        self.color_active = colour_pallete['white']
        self.color = self.color_inactive
        self.pwd = pwd
        self.placeholder = placeholder
        self.text = text
        self.font = pg.font.Font(None, 32)
        self.txt_surface = self.font.render(text, True, self.color)
        self.active = False
        self.txt_surface = self.font.render(self.placeholder, True, self.color)
    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
            self.color = self.color_active if self.active else self.color_inactive
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    if self.text == self.pwd:
                        print("Correct Password")
                        return True
                    self.text = ''
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)
        if not self.active:
            self.txt_surface = self.font.render(self.placeholder, True, self.color)

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        pg.draw.rect(screen, self.color, self.rect, 2)