import pygame as pg
from pygame.locals import *
from ComponentModule.components import *
from colours import colour_palette
import time
from set_up_track import LoadTrack, ValidateTrack, ProcessTrack, OrderOfOperations
import numpy as np


def left_panel(screen_shape, left_panel_border_x):
    width = int(left_panel_border_x*0.85)
    height = width*0.3
    x_margin = int((left_panel_border_x - width)//2)
    reset_btn = Button(x_margin, int(0.9*screen_shape[1]), int(left_panel_border_x*0.85), height, "Reset")
    return reset_btn

def right_panel(background, throttle_val, brake_val, right_panel_border_x, screen_shape, font_size):
    x_margin = int(right_panel_border_x + (screen_shape[0]//128))
    panel_width = int(screen_shape[0] - right_panel_border_x)
    panel_height = screen_shape[1]
    slider_width, slider_height = int(panel_width* 15/16), int(panel_height * 0.03)
    #throttle
    throttle_text = Text(x_margin, (panel_height // 6), "Throttle", font_size) 
    throttle_text.draw(background)
    throttle_slider = Slider(x_margin, int(panel_height * 0.2), slider_width, slider_height, 0, 1, throttle_val, "Throttle", False)

    #brake
    brake_text = Text(x_margin, int(panel_height * 4/15), "Brake", font_size) 
    brake_text.draw(background)
    brake_slider = Slider(x_margin, int(panel_height * 0.3), slider_width, slider_height, 0, 1, brake_val, "Brake", False)
    return throttle_slider, brake_slider


class TopPanel:
    def __init__(self, background, screen_shape, font_size):
        self.end_x = int(0.8 * screen_shape[0])
        self.even_spacing = int(self.end_x // 6)
        self.x_margin = 0 # screen_shape[0]//128
        self.y = 60
        self.font_size = font_size
        self.background = background

    def time_display(self, time):
        time_text = Text(self.x_margin + (self.even_spacing * 5), self.y, f"Time: {time:.2f}", self.font_size)
        time_text.draw(self.background)

    def lap_display(self, lap_no):
        lap_text = Text(self.x_margin + (self.even_spacing * 4), self.y, f"Lap: {lap_no}", self.font_size) 
        lap_text.draw(self.background)
    
    def pb_display(self, pb):
        pb_text = Text(self.x_margin + (self.even_spacing * 3), self.y, f"PB: {pb}", self.font_size)
        pb_text.draw(self.background)

    def weather_display(self, weather):
        weather_text = Text(self.x_margin + self.even_spacing, self.y, f"Weather: {weather}", self.font_size) #temp
        weather_text.draw(self.background)
    
    def track_name_display(self, track_name, track_list):
        track_dropdown = Dropdown(self.x_margin + (self.even_spacing * 2), self.y, 150, 30, track_name, track_list)
        return track_dropdown


       
        
def main():
    # Initialise screen
    pg.init()
    start_time = time.time()

    screen = pg.display.set_mode((1920, 1080))
    screen_shape = screen.get_size()
    pg.display.set_caption('Racing Lines')

    # Fill background
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill(colour_palette['BG_GREY'].value)

    # Display some text
    font_size = screen_shape[1]//30
    font = pg.font.Font(None, font_size)


    
    right_panel_border_x = 0.8 * screen_shape[0]
    top_panel_border_y = screen_shape[1] // 6
    left_panel_border_x = 0.1 * screen_shape[0]

    divididers = [divider(0, top_panel_border_y, right_panel_border_x, 3), divider(right_panel_border_x, 0, 3, screen_shape[1]), divider(left_panel_border_x, top_panel_border_y, 3, screen_shape[1]-top_panel_border_y), divider(right_panel_border_x, 750, 383, 3)]   
    track_list = ["Monaco", "Silverstone", "Spa", "Interlagos", "Suzuka", "Yas Marina", "Import"] #will be updated with customs tracks that are added

    top_panel = TopPanel(background, screen_shape, font_size)

    track_name = "Silverstone"
    track_dropdown = top_panel.track_name_display("Silverstone", track_list)
    order_of_operations = OrderOfOperations(track_name, dark_mode=True)
    order_of_operations.run()
    track_image = order_of_operations.get_track_image()


    throttle_slider, brake_slider =  right_panel(background, 0.5, 0.9, right_panel_border_x, screen_shape, font_size)
    reset_btn = left_panel(screen_shape, left_panel_border_x)

    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            # if event.type == pg.MOUSEBUTTONDOWN:
            #     print(pg.mouse.get_pos())
        screen.blit(background, (0, 0))
        for d in divididers:
            d.draw(screen)
        
        #left panel
        reset_btn.handle_event(event)
        reset_btn.draw(screen, font)

        #right panel
        throttle_slider.draw(screen, font)
        brake_slider.draw(screen, font)

        #top panel
        top_panel.time_display(time.time() - start_time)
        top_panel.lap_display(1)
        top_panel.pb_display("0:00.000")
        top_panel.weather_display("Sunny")

        track_dropdown.handle_event(event)
        if track_name != track_dropdown.get_track():
            order_of_operations = OrderOfOperations(track_name, dark_mode=True)
            order_of_operations.run()
            track_image = order_of_operations.get_track_image()
        track_name = track_dropdown.get_track()
        track_dropdown.draw(screen, font)



        screen.blit(track_image, (500,300))
        
        pg.display.flip()


if __name__ == '__main__': main()
