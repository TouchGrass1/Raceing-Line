import pygame as pg
from pygame.locals import *
from ComponentModule.components import *
from colours import colour_palette
import time
from set_up_track import OrderOfOperations
import numpy as np
from ComponentModule.zoom import *
from ComponentModule.clickNdrag import *


# ---------- PANEL BASE CLASS ----------
class BasePanel:
    def __init__(self, screen_shape, font, font_size):
        self.screen_shape = screen_shape
        self.font = font
        self.font_size = font_size

    def _draw_text(self, surface, text, x, y, colour="WHITE"):
        text_surface = self.font.render(text, True, colour_palette[colour].value)
        surface.blit(text_surface, (x, y))


# ---------- TOP PANEL ----------
class TopPanel(BasePanel):
    def __init__(self, screen_shape, font, font_size):
        super().__init__(screen_shape, font, font_size)
        self.end_x = int(0.8 * screen_shape[0])
        self.even_spacing = int(self.end_x // 6)
        self.x_margin = 0
        self.y = 60
        self.screen_shape = screen_shape

    def time_display(self, surface, time_val):
        self._draw_text(surface, f"Time: {time_val:.2f}", self.x_margin + (self.even_spacing * 5), self.y)

    def lap_display(self, surface, lap_no):
        self._draw_text(surface, f"Lap: {lap_no}", self.x_margin + (self.even_spacing * 4), self.y)

    def pb_display(self, surface, pb):
        self._draw_text(surface, f"PB: {pb}", self.x_margin + (self.even_spacing * 3), self.y)

    def weather_display(self, surface, weather):
        self._draw_text(surface, f"Weather: {weather}", self.x_margin + self.even_spacing, self.y)

    def track_name_display(self, track_name, track_list):
        return Dropdown(self.x_margin + (self.even_spacing * 2), self.y, self.screen_shape, track_name, track_list)


# ---------- LEFT PANEL ----------
class LeftPanel(BasePanel):
    def __init__(self, screen_shape, font, font_size):
        super().__init__(screen_shape, font, font_size)
        self.border_x = int(0.1 * screen_shape[0])
        width = int(self.border_x * 0.85)
        height = width * 0.3
        x_margin = int((self.border_x - width) // 2)
        y_pos = int(0.9 * screen_shape[1])
        self.reset_btn = Button(x_margin, y_pos, screen_shape, "Reset")

    def draw(self, surface):
        self.reset_btn.draw(surface, self.font)

    def handle_event(self, event):
        self.reset_btn.handle_event(event)


# ---------- RIGHT PANEL ----------
class RightPanel(BasePanel):
    def __init__(self, screen_shape, font, font_size, throttle_val=0.5, brake_val=0.9):
        super().__init__(screen_shape, font, font_size)
        self.border_x = int(0.8 * screen_shape[0])
        x_margin = int(self.border_x + (screen_shape[0] // 128))
        panel_width = int(screen_shape[0] - self.border_x)
        panel_height = screen_shape[1]
        slider_width, slider_height = int(panel_width * 15/16), int(panel_height * 0.03)

        # throttle slider
        self.throttle_slider = Slider(x_margin, int(panel_height * 0.2), slider_width, slider_height, 0, 1, throttle_val, "Throttle", False)
        self.throttle_pos = (x_margin, panel_height // 6)

        # brake slider
        self.brake_slider = Slider(x_margin, int(panel_height * 0.3), slider_width, slider_height, 0, 1, brake_val, "Brake", False)
        self.brake_pos = (x_margin, int(panel_height * 4/15))

    def draw(self, surface):
        self._draw_text(surface, "Throttle", *self.throttle_pos)
        self._draw_text(surface, "Brake", *self.brake_pos)
        self.throttle_slider.draw(surface, self.font)
        self.brake_slider.draw(surface, self.font)




# ---------- MAIN GAME ----------
def main():
    # Initialise screen
    pg.init()
    start_time = time.time()

    screen = pg.display.set_mode(flags=pg.FULLSCREEN)
    #screen = pg.display.set_mode((720,640))
    screen_shape = screen.get_size()
    pg.display.set_caption('Racing Lines')

    # Background surface (static)
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill(colour_palette['BG_GREY'].value)

    # Font
    font_size = screen_shape[1]//30
    font = pg.font.Font(None, font_size)

    # Panels
    top_panel = TopPanel(screen_shape, font, font_size)
    left_panel = LeftPanel(screen_shape, font, font_size)
    right_panel = RightPanel(screen_shape, font, font_size)

    # Track setup
    track_list = ["Monaco", "Silverstone", "Spa", "Interlagos", "Suzuka", "Yas Marina", "Import"]
    track_name = "Silverstone"
    track_dropdown = top_panel.track_name_display(track_name, track_list)

    order_of_operations = OrderOfOperations(track_name, dark_mode=True)
    order_of_operations.run()
    track_image = order_of_operations.get_track_image()

    # Clipping rectangle for track image
    x = int(0.1 * screen_shape[0])
    y = screen_shape[1] // 6
    w = int(0.8 * screen_shape[0]) - x
    h = screen_shape[1] - y
    DIVIDER_THICKNESS = 3
    clip_rect = pg.Rect(x + DIVIDER_THICKNESS, y + DIVIDER_THICKNESS, w - DIVIDER_THICKNESS, h - DIVIDER_THICKNESS)
    track_image_rect = track_image.get_rect()
    track_image_rect.center = (x+w//2, y+h//2)

    # Dividers
    dividers = Dividers(screen_shape)

    #Drag and Zoom
    zoom = Zoom(track_image)
    drag = Drag()



    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            if event.type == MOUSEBUTTONDOWN:
                print(pg.mouse.get_pos())

            if event.type == MOUSEWHEEL:
                zoom.handle_event(event)

            left_panel.handle_event(event)
            track_dropdown.handle_event(event)

        # redraw background
        screen.blit(background, (0, 0))

        # draw dividers
        dividers.draw(screen)

        # left panel
        left_panel.draw(screen)

        # right panel
        right_panel.draw(screen)

        # top panel overlays
        top_panel.time_display(screen, time.time() - start_time)
        top_panel.lap_display(screen, 1)
        top_panel.pb_display(screen, "0:00.000")
        top_panel.weather_display(screen, "Sunny")

        # track dropdown
        if track_name != track_dropdown.get_track():
            order_of_operations = OrderOfOperations(track_name, dark_mode=True)
            order_of_operations.run()
            track_image = order_of_operations.get_track_image()
            if track_image is not None:
                track_name = track_dropdown.get_track()
                track_image_rect = track_image.get_rect()
                track_image_rect.center = (x+w//2, y+h//2)
                zoom = Zoom(track_image)
        track_dropdown.draw(screen, font)


        screen.set_clip(clip_rect)

        # draw track image
        img = zoom.get_image()
        rect = img.get_rect()
        track_image_rect.topleft = drag.handle_event(event, rect, track_image_rect.topleft)
        screen.blit(track_image, track_image_rect.topleft)

        # reset clip
        screen.set_clip(None)

        pg.display.flip()


if __name__ == '__main__':
    main()
