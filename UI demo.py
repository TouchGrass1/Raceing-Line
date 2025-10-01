import pygame as pg
from pygame.locals import *
from ComponentModule.components import *
from colours import colour_palette




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
    brake_text = Text(x_margin, int(panel_height * 4/15), "brake", font_size) 
    brake_text.draw(background)
    brake_slider = Slider(x_margin, int(panel_height * 0.3), slider_width, slider_height, 0, 1, brake_val, "Brake", False)
    return throttle_slider, brake_slider

def top_panel(background, time, lap_no, pb, track_name, weather, screen_shape, font_size, track_list):
    end_x = int(0.8 * screen_shape[0])
    even_spacing = int(end_x // 6)
    x_margin = 0 # screen_shape[0]//128
    y = 60
    time_text = Text(x_margin + (even_spacing * 5) , y, f"Time: {time}", font_size)
    time_text.draw(background)

    lap_text = Text(x_margin + (even_spacing * 4), y, f"Lap: {lap_no}", font_size) 
    lap_text.draw(background)

    pb_text = Text(x_margin + (even_spacing * 3), y, f"PB: {pb}", font_size)
    pb_text.draw(background)

    track_text = Text(x_margin + (even_spacing * 2), y, f"Track: {track_name}", font_size) #temp
    track_text.draw(background)
    track_dropdown = Dropdown(x_margin + (even_spacing * 2), y + 40, 150, 30, track_name, track_list)

    weather_text = Text(x_margin + even_spacing, y, f"Weather: {weather}", font_size) #temp
    weather_text.draw(background)

    return track_dropdown


       
        
def main():
    # Initialise screen
    pg.init()
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
    track_list = ["Monaco", "Silverstone", "Spa", "Interlagos", "Suzuka", "Yas Marina", "Import"]#will be updated with customs tracks that are added
    track_dropdown = top_panel(background, "0:00.000", 1, "0:00.000", "Monaco", "Sunny", screen_shape, font_size, track_list)
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
            
        reset_btn.handle_event(event)
        reset_btn.draw(screen, font)
        throttle_slider.draw(screen, font)
        brake_slider.draw(screen, font)
        track_dropdown.handle_event(event)
        track_dropdown.draw(screen, font)
        
        pg.display.flip()


if __name__ == '__main__': main()
