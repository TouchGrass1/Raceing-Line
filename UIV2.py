import pygame as pg
from pygame.locals import *
from ComponentModule.components import *
from colours import colour_palette
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np

from TrackProcessing2.genetic_algorithm import main as ga
from TrackProcessing2.config import config, real_properties, default_variables, variable_options


# ---------- PANEL BASE CLASS ----------
class BasePanel:
    def __init__(self, screen_shape, font):
        self.screen_shape = screen_shape
        self.font = font

    def _draw_text(self, surface, text, x, y, colour="WHITE"):
        text_surface = self.font.render(text, True, colour_palette[colour].value)
        surface.blit(text_surface, (x, y))


# ---------- TOP PANEL ----------
class TopPanel(BasePanel):
    def __init__(self, screen_shape, font):
        super().__init__(screen_shape, font)
        self.end_x = int(0.8 * screen_shape[0])
        self.even_spacing = int(self.end_x // 6)
        self.x_margin = 0
        self.y = 60
        self.screen_shape = screen_shape
        self.weather_toggle = Toggle((self.x_margin + self.even_spacing), self.y, self.screen_shape, variable_options['weather'])
        self.show_popup = False


    def time_display(self, surface, time_val):
        self._draw_text(surface, f"Time: {time_val:.2f}", self.x_margin + (self.even_spacing * 5), self.y)

    def lap_display(self, surface, lap_no):
        self._draw_text(surface, f"Lap: {lap_no:.2f}", self.x_margin + (self.even_spacing * 4), self.y)

    def pb_display(self, surface, pb):
        self._draw_text(surface, f"PB: {pb}", self.x_margin + (self.even_spacing * 3), self.y)

    def weather_display(self, surface):
        self.weather_toggle.toggle_draw(surface, self.font)

    def track_name_display(self, track_name, track_list):
        return Dropdown(self.x_margin + (self.even_spacing * 2), self.y, self.screen_shape, track_name, track_list)
    
    def handle_event(self, event):
        self.weather_toggle.change_state(event)
    
    def draw_popup(self, surface):
        #draw an overlay to dim screen
        overlay = pg.Surface(self.screen_shape, pg.SRCALPHA)
        overlay.fill((0, 0, 0, 180)) 
        surface.blit(overlay, (0, 0))

        #draw base panel
        #self.popup_rect = pg.Rect((self.screen_shape[0] // 6), (self.screen_shape[1] // 6), (2*self.screen_shape[0]) // 3, (2*self.screen_shape[1]) // 3 )
        #pg.draw.rect(surface, colour_palette["SUBTLE_GREY"].value, (0,0, 300, 500))

        #draw components
        self._draw_text(surface, "Calculating Racing Line", self.screen_shape[0]//2, self.screen_shape[1]//2)

    def handle_popup_events(self, event):             
        #close if click outside
        if event.type == pg.MOUSEBUTTONDOWN:
            if not self.popup_rect.collidepoint(event.pos): self.show_import_popup = False

        return True



# ---------- LEFT PANEL ----------
class LeftPanel(BasePanel):
    def __init__(self, screen_shape, font):
        super().__init__(screen_shape, font)
        self.border_x = int(0.1 * screen_shape[0])
        width = int(self.border_x * 0.85)
        height = width * 0.3
        x_margin = int((self.border_x - width) // 2)
        self.even_spacing = int(width // 6)
        

        self.reset_btn = Button(x_margin, screen_shape[1] - 3*self.even_spacing, screen_shape, "Reset")
        self.pause_toggle = Toggle(x_margin, screen_shape[1] - 6*self.even_spacing, screen_shape, ["Pause", "Play"])

    def draw(self, surface):
        self.reset_btn.draw(surface, self.font)
        self.pause_toggle.toggle_draw(surface, self.font)


    def handle_event(self, event):
        self.reset_btn.handle_event(event)
        self.pause_toggle.change_state(event)


# ---------- RIGHT PANEL ----------
class RightPanel(BasePanel):
    def __init__(self, screen_shape, font, small_font):
        super().__init__(screen_shape, font)
        self.small_font = small_font
        self.border_x = int(0.8 * screen_shape[0])
        self.x_margin = int(self.border_x + (screen_shape[0] // 128))
        panel_width = int(screen_shape[0] - self.border_x)
        self.panel_height = screen_shape[1]
        self.screen_shape = screen_shape
        throttle_val=0.45
        brake_val=0.9
        fuel_val = 0.1
        tyre_wear_val = 0.99
        self.window_size = 10
        
        self.slider_width, self.slider_height = int(panel_width * 15/16), int(self.panel_height * 0.03)
        self.slider_width_small, self.slider_height_small = int(panel_width * 10/16), int(self.panel_height * 0.02)
        # throttle slider
        self.throttle_slider = Slider(self.x_margin, int(self.panel_height * 0.2), self.slider_width, self.slider_height, 0, 1, throttle_val, "Throttle", False)
        self.throttle_pos = (self.x_margin, self.panel_height // 6)

        # brake slider
        self.brake_slider = Slider(self.x_margin, int(self.panel_height * 0.3), self.slider_width, self.slider_height, 0, 1, brake_val, "Brake", False)
        self.brake_pos = (self.x_margin, int(self.panel_height * 4/15))

        # tyre toggle
        self.tyre_toggle = Toggle((self.border_x + (panel_width//3)), int(self.panel_height * 0.35), self.screen_shape, variable_options['tyre'])

        #fuel slider
        self.fuel_slider = Slider(1.04*self.x_margin, int(self.panel_height * 0.8), self.slider_width_small, self.slider_height_small, 0, 1, fuel_val, "Fuel", True)
        #tyre wear slider
        self.tyreWear_slider = Slider(1.04*self.x_margin, int(self.panel_height * 0.9), self.slider_width_small, self.slider_height_small, 0, 1, tyre_wear_val, "Tyre Wear", True)

        # Graphs
        self.speed_graph = TelemetryGraph((self.border_x + 5*(self.x_margin - self.border_x)), int(self.panel_height * 0.42), (panel_width - 10*(self.x_margin - self.border_x)), (panel_width - 10*(self.x_margin - self.border_x)), "Speed vs Time")
        self.GForce_graph = TelemetryGraph((self.border_x + 13*(self.x_margin - self.border_x)), int(self.panel_height * 0.5), (panel_width - 20*(self.x_margin - self.border_x)), (panel_width - 20*(self.x_margin - self.border_x)), "G Force")

        #Window control
        self.window_input = EntryBox(self.x_margin, int(self.panel_height * 0.65), (panel_width - 2*(self.x_margin - self.border_x)), 50, placeholder='Enter Window Size', is_password=False)

    def update_health_bar_sliders(self, throttle_val, brake_val,):
        self.throttle_slider = Slider(self.x_margin, int(self.panel_height * 0.2), self.slider_width, self.slider_height, 0, 1, throttle_val, "Throttle", False)
        self.brake_slider = Slider(self.x_margin, int(self.panel_height * 0.3), self.slider_width, self.slider_height, 0, 1, brake_val, "Brake", False)

    def handle_event(self, event):
        self.tyre_toggle.change_state(event)
        self.fuel_slider.listen(event)
        self.tyreWear_slider.listen(event)
        if self.window_input.handle_event(event):
            self.window_size = int(self.window_input.get_text())
        
    def draw(self, surface, sim_time):
        self._draw_text(surface, "Throttle", *self.throttle_pos)
        self._draw_text(surface, "Brake", *self.brake_pos)
        self.throttle_slider.draw(surface, self.font)
        self.brake_slider.draw(surface, self.font)
        self.tyre_toggle.toggle_draw(surface, self.font)
        self.fuel_slider.draw(surface, self.font)
        self.tyreWear_slider.draw(surface, self.font)
        self.speed_graph.draw(surface, self.small_font, sim_time, self.window_size)
        self.window_input.draw(surface)
        #self.GForce_graph.draw(surface, self.small_font, sim_time)
        return


# ---------- Center PANEL ----------
class CenterPanel(BasePanel):
    def __init__(self, screen_shape, font):
        super().__init__(screen_shape, font)
        self.start_x = int(0.1 * screen_shape[0]) + 3 #divider thickness
        self.width = int(0.7 * screen_shape[0]) -3
        self.y = (screen_shape[1] // 6) + 3
        self.height = screen_shape[1] - self.y
        self.y_padding = self.height // 10
        self.even_spacing = int(self.width // 5)

        self.zoom_in = Button( self.start_x + self.even_spacing*4, (screen_shape[1] - self.y_padding), screen_shape, "+", 'circle')
        self.zoom_out= Button( (self.start_x + self.even_spacing*4 + 2*(30/1080 * screen_shape[1])), (screen_shape[1] - self.y_padding), screen_shape, "-", 'circle')
    5
    
    def draw_zoom_btns(self, surface, font):
        self.zoom_in.draw(surface, font)
        self.zoom_out.draw(surface, font)
    
    def handle_event(self, event):
        self.zoom_in.handle_event(event)
        self.zoom_out.handle_event(event)

class Drag:
    def __init__(self):
        self.is_dragging = False
        self.drag_start_mouse = (0, 0)
        self.drag_start_pan = (0, 0)

    def update(self, event, clip_rect, current_pan):
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
            if clip_rect.collidepoint(event.pos):
                self.is_dragging = True
                self.drag_start_mouse = event.pos
                self.drag_start_pan = current_pan

        elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False

        elif event.type == pg.MOUSEMOTION and self.is_dragging:
            dx = event.pos[0] - self.drag_start_mouse[0]
            dy = event.pos[1] - self.drag_start_mouse[1]
            #calculate new pan based on movement since click
            return (self.drag_start_pan[0] + dx, self.drag_start_pan[1] + dy)

        return current_pan
    
def reset_sim():
    start_time = time.time()
    variables = default_variables.copy()
    return start_time, variables

def open_file_browser():
    root = tk.Tk()
    root.withdraw() 
    root.attributes("-topmost", True) #bring to top
    
    file_path = filedialog.askopenfilename(
        title="Select Track Image",
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp"), ("All Files", "*.*")]
    )
    
    if file_path == "":
        return None, None

    props_path = filedialog.askopenfilename(
            title="Select Track Properties (.txt)",
            filetypes=[("Text Files", "*.txt")]
        )
    root.destroy()
    
    return file_path, props_path

def add_real_properties(track_name, track_properties_path):
    new_props = {
        'real_track_length': 1000, 
        'real_track_width': 12
    }
    with open(track_properties_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, value = line.split(':')   
                new_props[key.strip()] = float(value.strip()) # remove whitespace and convert to float

        real_properties[track_name] = new_props
        print(f"Properties updated for {track_name}: {new_props}")

def run_GA(variables, clip_rect):
    print(f"Running GA for {variables['track']}...")
            
    racing_line, best_time, vels, mesh = ga(variables)
    
    all_x = mesh[:, :, 0]
    all_y = mesh[:, :, 1]
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    
    #store the track bounds as pygame Rect
    track_rect = pg.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    
    scale, offset = calculate_auto_scale(mesh, clip_rect)
    pb_str = f"{int(best_time[-1]//60)}: {best_time[-1]%60:.3f}"
    start_time = time.time()
    acceleration = caculateAccelerations(vels, best_time)
    return racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration

def zoom(track_rect, clip_rect, scale, in_bool):
    zoom_step = 1.3 if in_bool else 0.7
    scale = np.clip(scale * zoom_step, 0.01, 50.0)
  
    offset_x = clip_rect.centerx - (track_rect.centerx * scale)
    offset_y = clip_rect.centery - (track_rect.centery * scale)
    
    return scale, (offset_x, offset_y)

def draw_track_elements(screen, mesh, racing_line, velocities,  scale, offset):

    left_boundary = mesh[:, 0, :]
    right_boundary = mesh[:, -1, :]
    
    left_pts = [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1])) for p in left_boundary]
    right_pts = [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1])) for p in right_boundary]
    
    pg.draw.lines(screen, colour_palette['RED'].value, True, left_pts, 2)
    pg.draw.lines(screen, colour_palette['GREEN'].value, True, right_pts, 2)
    
    #racing line
    max_v = np.max(velocities)
    min_v = np.min(velocities)

    for i in range(len(racing_line) - 1):
            p1 = racing_line[i]
            p2 = racing_line[i+1]
   
            norm_v = (velocities[i] - min_v) / (max_v - min_v) #normalise
            color = (int(255 *  norm_v), int(255 * (1-norm_v)), 0) #LERP BETWEEN RED and GREEN
            
            start_pos = (int(p1[0] * scale + offset[0]), int(p1[1] * scale + offset[1]))
            end_pos = (int(p2[0] * scale + offset[0]), int(p2[1] * scale + offset[1]))
            
            pg.draw.line(screen, color, start_pos, end_pos, 4)

def calculate_auto_scale(mesh, clip_rect, padding=40):
    # Extract all X and Y coordinates from the mesh
    all_x = mesh[:, :, 0].flatten()
    all_y = mesh[:, :, 1].flatten()
    
    # Find the bounds of the track
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    
    track_w = max_x - min_x
    track_h = max_y - min_y
    
    # Available space in the UI (from your clip_rect)
    available_w = clip_rect.width - (padding * 2)
    available_h = clip_rect.height - (padding * 2)
    
    # Determine the limiting scale factor
    scale = min(available_w / track_w, available_h / track_h)
    
    # Calculate offset to center the track in the clip_rect
    offset_x = clip_rect.x + padding - (min_x * scale) + (available_w - track_w * scale) / 2
    offset_y = clip_rect.y + padding - (min_y * scale) + (available_h - track_h * scale) / 2
    
    return scale, (offset_x, offset_y)

def get_car_position(sim_time, racing_line, time_array, scale, offset):


    racing_line = np.vstack([racing_line, racing_line[0]])
    car_x = np.interp(sim_time, time_array, racing_line[:, 0]) #LERP the car pos
    car_y = np.interp(sim_time, time_array, racing_line[:, 1])

    car_x = car_x* scale + offset[0]
    car_y = car_y* scale + offset[1]

    diffs = np.diff(racing_line, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]) #find angles of heading in RAD
    angles = np.append(angles, angles[0])
    
    car_angle = np.interp(sim_time, time_array, angles)
    
    return (car_x, car_y), car_angle

def draw_car(screen, pos, angle):
    car_length = 20
    car_width = 18

    pts = [
        (car_length // 2, 0),               # Front tip
        (-car_length // 2, -car_width // 2), # Back left
        (-car_length // 2, car_width // 2)   # Back right
    ]
    rotated_pts = []
    for x, y in pts:
        rx = x * np.cos(angle) - y * np.sin(angle)
        ry = x * np.sin(angle) + y * np.cos(angle)
        rotated_pts.append((pos[0] + rx, pos[1] + ry))

    pg.draw.polygon(screen, colour_palette['ORANGE'].value, rotated_pts)

def caculateAccelerations(vel, t):     
    a_arr = np.zeros(len(vel))
    for i in range(1, len(vel)):
        dt = t[i] - t[i-1]
        if dt == 0:
            a_arr[i] = 0
        else:
            
            a_arr[i] = (vel[i] - vel[i-1]) / dt
    return a_arr

def get_currentThrottle(sim_time, time_array, acceleration_array):
    current_acceleration = np.interp(sim_time, time_array[:-1], acceleration_array)
    if current_acceleration > 0:
        current_throttle = 1

    else: current_throttle = 0
    return current_throttle
# ---------- MAIN GAME ----------
def main():
    # Initialise screen
    pg.init()
    clock = pg.time.Clock()
    start_time = time.time()

    screen = pg.display.set_mode(flags=pg.FULLSCREEN)
    screen = pg.display.set_mode((1920,1080))
    screen_shape = screen.get_size()
    pg.display.set_caption('Racing Lines')

    #Track data
    racing_line, best_time, vels, mesh = None, None, None, None
    pb_str = "00:00.000"
    best_time = [1]
    paused = False
    pause_start_time = 0
    total_paused_duration = 0


    # Background surface (static)
    background = pg.Surface(screen.get_size())
    background = background.convert()
    background.fill(colour_palette['BG_GREY'].value)

    # Font
    font_size = screen_shape[1]//30
    font = pg.font.Font(None, font_size)
    small_font = pg.font.Font(None, font_size // 2)

    scale = 1

    # Panels
    top_panel = TopPanel(screen_shape, font)
    left_panel = LeftPanel(screen_shape, font)
    right_panel = RightPanel(screen_shape, font, small_font)
    center_panel = CenterPanel(screen_shape, font)



    # Track setup
    #track_list = ["Monaco", "Silverstone", "Spa", "Interlagos", "Suzuka", "Yas Marina", "Import"]
    variables = default_variables.copy()
    track_dropdown = top_panel.track_name_display(variables['track'], variable_options['track'])

    # Clipping rectangle for track image
    x = int(0.1 * screen_shape[0])
    y = screen_shape[1] // 6
    w = int(0.8 * screen_shape[0]) - x
    h = screen_shape[1] - y
    DIVIDER_THICKNESS = 3
    clip_rect = pg.Rect(x + DIVIDER_THICKNESS, y + DIVIDER_THICKNESS, w - DIVIDER_THICKNESS, h - DIVIDER_THICKNESS)
    pan = (0, 0)
    dragger = Drag()
    track_rect = None
    # Dividers
    dividers = Dividers(screen_shape)



    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
                      
            pan = dragger.update(event, clip_rect, pan)
            if event.type == MOUSEBUTTONDOWN:
                print(pg.mouse.get_pos())
                if left_panel.reset_btn.rect.collidepoint(event.pos):
                    start_time, variables = reset_sim()
                    pan = (0, 0)
                    scale, offset = calculate_auto_scale(mesh, clip_rect)

                if center_panel.zoom_in.rect.collidepoint(event.pos):
                    scale, offset = zoom(track_rect, screen.get_clip(), scale, True)
                if center_panel.zoom_out.rect.collidepoint(event.pos):
                    scale, offset = zoom(track_rect, screen.get_clip(), scale, False)
                if left_panel.pause_toggle.rect.collidepoint(event.pos):
                    paused = not paused
                    if paused:
                        pause_start_time = time.time()
                    else:
                        total_paused_duration += time.time() - pause_start_time

            left_panel.handle_event(event)
            track_dropdown.handle_event(event)
            right_panel.handle_event(event)
            top_panel.handle_event(event)
            center_panel.handle_event(event)

        if variables['tyre'] != right_panel.tyre_toggle.get_state():
            variables['tyre'] = right_panel.tyre_toggle.get_state()
            racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration = run_GA(variables, clip_rect)

        # track dropdown
        if variables['track'] != track_dropdown.get_track():
            new_selection = track_dropdown.get_track()
            
            if new_selection == 'import':
                path, props_path = open_file_browser()
                if path:
                    track_filename = Path(path).stem
                    add_real_properties(track_filename, props_path)
                    track_dropdown.update_options(track_filename)

                    variables['track'] = track_filename
                    variables['custom_path'] = path
                    pan = (0,0)
                    top_panel.show_popup = True
                    top_panel.draw_popup(screen)
                    pg.display.flip()       
                    racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration = run_GA(variables, clip_rect)
                else:
                    track_dropdown.set_track(variables['track'])
                    print("Import cancelled")
            else:
                variables['track'] = new_selection
                variables['custom_path'] = None
                pan = (0,0)
                top_panel.show_popup = True
                top_panel.draw_popup(screen)
                pg.display.flip()        
                racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration = run_GA(variables, clip_rect)
            
            right_panel.speed_graph.precompute_telemetry(best_time, vels)
            right_panel.GForce_graph.precompute_telemetry(best_time, acceleration)
            top_panel.show_popup = False
            
            print("----")
            print(f"Track changed to: {variables['track']}")
            print("----")

        if not paused: 
            current_time = (time.time() - start_time) - total_paused_duration
        sim_time = current_time % best_time[-1]


        
        #DRAWING
        screen.blit(background, (0, 0))
        dividers.draw(screen)
        left_panel.draw(screen)
        right_panel.draw(screen, sim_time)
        

        # top panel overlays
        top_panel.time_display(screen, current_time)
        top_panel.lap_display(screen, current_time / best_time[-1])
        top_panel.pb_display(screen, pb_str)
        top_panel.weather_display(screen)
        track_dropdown.draw(screen, font)

        screen.set_clip(clip_rect)
        if mesh is not None:
            render_offset = (offset[0] + pan[0], offset[1] + pan[1])
            draw_track_elements(screen, mesh, racing_line, vels, scale, render_offset)
            pos, angle = get_car_position(sim_time, racing_line, best_time, scale, render_offset)
            draw_car(screen, pos, angle)
            throttle = get_currentThrottle(sim_time, best_time, acceleration)
            right_panel.update_health_bar_sliders(throttle, (1-throttle))
            center_panel.draw_zoom_btns(screen, font)
        
        screen.set_clip(None)
        
        
        clock.tick(60)
        pg.display.flip()


if __name__ == '__main__':
    main()