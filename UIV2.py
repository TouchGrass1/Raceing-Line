import pygame as pg
from pygame.locals import *
from ComponentModule.components import *
from colours import colour_palette
import time
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import numpy as np

from TrackProcessing2.ga_v2 import main as ga
from TrackProcessing2.config import config, real_properties, default_variables, variable_options


# ---------- PANEL BASE CLASS ----------
class BasePanel:
    def __init__(self, screen_shape, font):
        self.screen_shape = screen_shape
        self.font = font
        DIVIDER_THICKNESS = 3

        #panel bounds
        self.top_panel_end_y = screen_shape[1] // 6
        self.right_panel_start_x = int(0.8 * screen_shape[0])
        self.left_panel_end_x = int(0.1 * screen_shape[0])
        self.right_pages_end_y = int(0.7 * screen_shape[1])
        self.right_pages_width = screen_shape[0] - self.right_panel_start_x


        #dividers
        self.dividers = {
            "top_panel": Divider(0, self.top_panel_end_y, self.right_panel_start_x, DIVIDER_THICKNESS),
            "right_panel": Divider(self.right_panel_start_x, 0, DIVIDER_THICKNESS, screen_shape[1]),
            "left_panel": Divider(self.left_panel_end_x, self.top_panel_end_y, DIVIDER_THICKNESS, screen_shape[1] - self.top_panel_end_y),
            "right_pages": Divider(self.right_panel_start_x, self.right_pages_end_y, self.right_pages_width, DIVIDER_THICKNESS),
        }


    def _draw_text(self, surface, text, x, y, colour="WHITE"):
        text_surface = self.font.render(text, True, colour_palette[colour].value)
        surface.blit(text_surface, (x, y))
    
    def draw_dividers(self, screen):
        for divider in self.dividers.values():
            divider.draw(screen)

    def get(self, name):
        return self.dividers.get(name)

class Divider:
    def __init__(self, x, y, w, h):
        self.rect = pg.Rect(x, y, w, h)

    def draw(self, screen):
        pg.draw.rect(screen, colour_pallete['line grey'], self.rect)

# ---------- TOP PANEL ----------
class TopPanel(BasePanel):
    def __init__(self, screen_shape, font):
        super().__init__(screen_shape, font)

        self.even_spacing = int(self.right_panel_start_x // 6)
        self.x_margin = 0
        self.y = 60 #level at which text is drawn

        self.weather_toggle = Toggle((self.x_margin + self.even_spacing), self.y, self.screen_shape, variable_options['weather'])
        self.show_popup = False


    def time_display(self, surface, time_val):
        self._draw_text(surface, f"Time: {int(time_val//60)}: {time_val%60:.2f}", self.x_margin + (self.even_spacing * 5), self.y)

    def lap_display(self, surface, lap_no):
        self._draw_text(surface, f"Lap: {lap_no:.2f}", self.x_margin + (self.even_spacing * 4), self.y)

    def pb_display(self, surface, pb):
        self._draw_text(surface, f"PB: {pb}", self.x_margin + (self.even_spacing * 3), self.y)

    def weather_text(self, surface):
        self._draw_text(surface, "Weather:", self.x_margin + (self.even_spacing * 0.54), self.y)

    def weather_display(self, surface):
        self.weather_toggle.draw(surface, self.font)

    def track_name_display(self, track_list):
        return Dropdown(self.x_margin + (self.even_spacing * 2), self.y, self.screen_shape, "Select Track", track_list)
    
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
        width = int(self.left_panel_end_x * 0.85)
        x_margin = int((self.left_panel_end_x - width) // 2)
        self.even_spacing = int(width // 6)
        
        self.ghost_line = None
        self.ghost_vels = None
        self.ghost_times = None

        

        self.reset_btn = Button(x_margin, screen_shape[1] - 3*self.even_spacing, screen_shape, "Reset")
        self.pause_toggle = Toggle(x_margin, screen_shape[1] - 6*self.even_spacing, screen_shape, ["Pause", "Play"])

        self.save_ghost_btn = Button(x_margin, screen_shape[1] - 9*self.even_spacing, screen_shape, "Save Ghost")
        self.show_ghost_toggle = Toggle(x_margin, screen_shape[1] - 12*self.even_spacing, screen_shape, ["Show Ghost", "Hide Ghost"])

        self.follow_car = Toggle(x_margin, screen_shape[1] - 15*self.even_spacing, screen_shape, ["Follow Car", "Stop Follow"])

    def draw(self, surface):
        self.reset_btn.draw(surface, self.font)
        self.pause_toggle.draw(surface, self.font)

        self.save_ghost_btn.draw(surface, self.font)
        self.show_ghost_toggle.draw(surface, self.font)

        self.follow_car.draw(surface, self.font)


    def handle_event(self, event):
        self.reset_btn.handle_event(event)
        self.pause_toggle.change_state(event)

        self.save_ghost_btn.handle_event(event)
        self.show_ghost_toggle.change_state(event)

        self.follow_car.change_state(event)


    def save_ghost_func(self, racing_line, vels, times):
        if self.save_ghost_btn.get_clicked():
            self.ghost_line = racing_line
            self.ghost_vels = vels
            self.ghost_times = times

    
    def reset(self):
        self.ghost_line = None
        self.ghost_vels = None
        self.ghost_times = None
        self.show_ghost_toggle.set_state(0)
        self.pause_toggle.set_state(0)

    
    def show_ghost_func(self, surface, scale, offset):
        if self.show_ghost_toggle.get_state() == "Hide Ghost" and self.ghost_line is not None: 
            max_v = np.max(self.ghost_vels)
            min_v = np.min(self.ghost_vels)

            racing_line = np.vstack([self.ghost_line, self.ghost_line[0]])


            for i in range(len(racing_line) - 1):
                    p1 = racing_line[i]
                    p2 = racing_line[i+1]
        
                    norm_v = (self.ghost_vels[i] - min_v) / (max_v - min_v) #normalise
                    color = (int(255 *  norm_v), 0, int(255 * (1-norm_v)), int(0.25*255)) #LERP BETWEEN RED and Blue, and 25% opacity
                    
                    start_pos = (int(p1[0] * scale + offset[0]), int(p1[1] * scale + offset[1]))
                    end_pos = (int(p2[0] * scale + offset[0]), int(p2[1] * scale + offset[1]))
                    
                    pg.draw.line(surface, color, start_pos, end_pos, 3)


# ---------- RIGHT PANEL ----------
class RightPanel(BasePanel):
    def __init__(self, screen_shape, font, small_font):
        super().__init__(screen_shape, font)
        self.small_font = small_font
        self.right_panel_start_x = int(0.8 * screen_shape[0])
        self.x_margin = int(self.right_panel_start_x + (screen_shape[0] // 128))
        panel_width = int(screen_shape[0] - self.right_panel_start_x)
        self.panel_height = screen_shape[1]

        #default values
        throttle_val=0.45
        brake_val=0.9
        fuel_val = 0.1
        lap_val = default_variables['lapNo']
        self.window_size = 10

        #components 
        self.slider_width, self.slider_height = int(panel_width * 15/16), int(self.panel_height * 0.03)
        self.slider_width_small, self.slider_height_small = int(panel_width * 10/16), int(self.panel_height * 0.02)
        # throttle slider
        self.throttle_slider = Slider(self.x_margin, int(self.panel_height * 0.2), self.slider_width, self.slider_height, 0, 1, throttle_val, "Throttle", False)
        self.throttle_pos = (self.x_margin, self.panel_height // 6)

        # brake slider
        self.brake_slider = Slider(self.x_margin, int(self.panel_height * 0.3), self.slider_width, self.slider_height, 0, 1, brake_val, "Brake", False)
        self.brake_pos = (self.x_margin, int(self.panel_height * 4/15))

        # tyre toggle
        self.tyre_toggle = Toggle((self.right_panel_start_x + (panel_width//3)), int(self.panel_height * 0.35), self.screen_shape, variable_options['tyre'])

        #fuel slider
        self.fuel_slider = Slider(1.04*self.x_margin, int(self.panel_height * 0.8), self.slider_width_small, self.slider_height_small, 0, 1, fuel_val, "Fuel", True)
        #tyre wear slider
        self.lap_no_slider = Slider(1.04*self.x_margin, int(self.panel_height * 0.9), self.slider_width_small, self.slider_height_small, 0, 70, lap_val, "Lap Number", True)

        # Graphs
        self.speed_graph = TelemetryGraph((self.right_panel_start_x + 5*(self.x_margin - self.right_panel_start_x)), int(self.panel_height * 0.42), (panel_width - 10*(self.x_margin - self.right_panel_start_x)), (panel_width - 10*(self.x_margin - self.right_panel_start_x)), "Speed vs Time")
        self.GForce_graph = TelemetryGraph((self.right_panel_start_x + 13*(self.x_margin - self.right_panel_start_x)), int(self.panel_height * 0.5), (panel_width - 20*(self.x_margin - self.right_panel_start_x)), (panel_width - 20*(self.x_margin - self.right_panel_start_x)), "G Force")

        #Window control
        self.window_input = EntryBox(self.x_margin, int(self.panel_height * 0.65), (panel_width - 2*(self.x_margin - self.right_panel_start_x)), (0.0429 * self.panel_height), self.font, placeholder='Change Graph Resolution', is_password=False)

        #recalculate
        self.recalculate_btn = Button((self.right_panel_start_x + (panel_width//3)), int(self.panel_height * 0.1), self.screen_shape, 'Recalculate')

    def update_health_bar_sliders(self, throttle_val, brake_val):
        self.throttle_slider = Slider(self.x_margin, int(self.panel_height * 0.2), self.slider_width, self.slider_height, 0, 1, throttle_val, "Throttle", False)
        self.brake_slider = Slider(self.x_margin, int(self.panel_height * 0.3), self.slider_width, self.slider_height, 0, 1, brake_val, "Brake", False)
        

    def handle_event(self, event):
        self.tyre_toggle.change_state(event)
        self.fuel_slider.handle_event(event)
        self.lap_no_slider.handle_event(event)
        self.window_input.handle_event(event)
        if self.window_input.is_correct == True: #if enter is pressed
            if self.window_input.get_text().isdigit(): #check if input is a number
                if int(self.window_input.get_text()) > 0 and int(self.window_input.get_text()) < 120: #check if number is in range
                    self.window_size = int(self.window_input.get_text())
        self.recalculate_btn.handle_event(event)
        
    def draw(self, surface, sim_time):
        self._draw_text(surface, "Throttle", *self.throttle_pos)
        self._draw_text(surface, "Brake", *self.brake_pos)
        self.throttle_slider.draw(surface, self.font)
        self.brake_slider.draw(surface, self.font)
        self.tyre_toggle.draw(surface, self.font)
        self.fuel_slider.draw(surface, self.font)
        self.lap_no_slider.draw(surface, self.font)
        self.speed_graph.draw(surface, self.small_font, sim_time, self.window_size)
        self.window_input.draw(surface)
        #self.GForce_graph.draw(surface, self.small_font, sim_time)
        self.recalculate_btn.draw(surface, self.font)
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
    #default values in case of missing file
    new_props = {
        'real_track_length': 1000, 
        'real_track_width': 12
        }  
    if track_properties_path == '':
        pass
    else:
        with open(track_properties_path, 'r') as f:
            for line in f:
                if ':' in line: #since file stores track properties as a dictionary
                    key, value = line.split(':')   
                    if float(value.strip()) < 1:
                        real_properties[track_name] = new_props #don't update if values are invalid
                        return
                    new_props[key.strip()] = float(value.strip()) # remove whitespace and convert to float

    real_properties[track_name] = new_props #update properites
    print(f"Properties updated for {track_name}: {new_props}")

def run_GA(variables, clip_rect):
    print(f"Running GA for {variables['track']}...")
            
    racing_line, best_time, vels, mesh = ga(variables)
    
    all_x = mesh[:, :, 0] #flatten x coords of mesh
    all_y = mesh[:, :, 1]
    min_x, max_x = np.min(all_x), np.max(all_x)
    min_y, max_y = np.min(all_y), np.max(all_y)
    
    #store the track bounds as pygame Rect
    track_rect = pg.Rect(min_x, min_y, max_x - min_x, max_y - min_y)
    
    scale, offset = calculate_auto_scale(clip_rect, track_rect)
    pb_str = f"{int(best_time[-1]//60)}: {best_time[-1]%60:.3f}"
    start_time = time.time()
    acceleration = caculateAccelerations(vels, best_time)
    return racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration

def zoom(current_scale, current_offset, in_bool, mouse_pos):
    zoom_step = 1.4

    if not in_bool: 
        zoom_step = 1 / zoom_step
    
    new_scale = np.clip(current_scale * zoom_step, 1, 75)

    actual_ratio = new_scale / current_scale #how much the scale is changing by, used to offset the zoom

    mx, my = mouse_pos
    ox, oy = current_offset

    #adjust offsets so the point under the mouse stays under the mouse
    new_offset_x = mx - (mx - ox) * actual_ratio
    new_offset_y = my - (my - oy) * actual_ratio

    
    return new_scale, (new_offset_x, new_offset_y)

def draw_track_elements(screen, mesh, racing_line, velocities,  scale, offset, mini_map_data=None):

    #draw boundaries
    left_pts = transform_pts(mesh[:, 0, :], scale, offset)
    right_pts = transform_pts(mesh[:, -1, :], scale, offset)
    
    pg.draw.lines(screen, colour_palette['RED'].value, True, left_pts, 2)
    pg.draw.lines(screen, colour_palette['GREEN'].value, True, right_pts, 2)
    
    #racing line    
    max_v = np.max(velocities)
    min_v = np.min(velocities)
    racing_line = np.vstack([racing_line, racing_line[0]])

    for i in range(len(racing_line) - 1):
        norm_v = (velocities[i] - min_v) / (max_v - min_v) #normalise
        color = (int(255 *  norm_v), int(255 * (1-norm_v)), 0) #LERP BETWEEN RED and GREEN
            
        start = transform_pts([racing_line[i]], scale, offset)[0]
        end = transform_pts([racing_line[i+1]], scale, offset)[0]
        pg.draw.line(screen, color, start, end, 4)
    
    if mini_map_data:
        m_rect = mini_map_data['rect']
        m_scale = mini_map_data['scale']
        m_off = mini_map_data['offset']        
    
        m_left = transform_pts(mesh[::15, 0, :], m_scale, m_off) #slice to reduce number of points drawn
        m_right = transform_pts(mesh[::15, -1, :], m_scale, m_off)
    
        #draw bg
        pg.draw.rect(screen, colour_palette['BG_GREY'].value, mini_map_data['rect'])
        pg.draw.rect(screen, colour_palette['WHITE'].value, m_rect, 1) #border

        #draw boundaries
        pg.draw.lines(screen, colour_palette['RED'].value, True, m_left, 1)
        pg.draw.lines(screen, colour_palette['GREEN'].value, True, m_right, 1)
        
        #draw car dot
        cx = int(mini_map_data['car_pos'][0] * m_scale + m_off[0])
        cy = int(mini_map_data['car_pos'][1] * m_scale + m_off[1])
        pg.draw.circle(screen, colour_palette['ORANGE'].value, (cx, cy), 4)

def calculate_auto_scale(clip_rect, track_rect, padding=40):     
    #central panel dimensions minus padding
    available_w = clip_rect.width - (padding * 2)
    available_h = clip_rect.height - (padding * 2)
    
    #calculate the scale to fit in center panel
    scale = min(available_w / track_rect.width, available_h / track_rect.height)
    
    #set track to the center of the panel
    offset_x = clip_rect.centerx - (track_rect.centerx * scale)
    offset_y = clip_rect.centery - (track_rect.centery * scale)
    
    return scale, (offset_x, offset_y)

def get_car_position(sim_time, racing_line, time_array):
    racing_line = np.vstack([racing_line, racing_line[0]])
    car_x = np.interp(sim_time, time_array, racing_line[:, 0]) #LERP the car pos
    car_y = np.interp(sim_time, time_array, racing_line[:, 1])

    diffs = np.diff(racing_line, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]) #find angles of heading in RAD
    angles = np.append(angles, angles[0])
    
    car_angle = np.interp(sim_time, time_array, angles)
    
    return (car_x, car_y), car_angle

def follow_car(sim_time, racing_line, time_array, scale, center):
    #set car to center of screen and adjust track acordingly
    racing_line = np.vstack([racing_line, racing_line[0]])
    car_x = np.interp(sim_time, time_array, racing_line[:, 0]) #LERP the car pos
    car_y = np.interp(sim_time, time_array, racing_line[:, 1])

    diffs = np.diff(racing_line, axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0]) #find angles of heading in RAD
    angles = np.append(angles, angles[0])
    
    car_angle = np.interp(sim_time, time_array, angles)

    new_offset = [center[0] - (car_x * scale), center[1] - (car_y * scale)] #makes the car the center of the screen



    return new_offset, center, car_angle

def draw_car(screen, pos, angle):
    car_length = 20
    car_width = 18
    #Triangle --> three points
    pts = [
        (car_length // 2, 0),               # Front tip
        (-car_length // 2, -car_width // 2), # Back left
        (-car_length // 2, car_width // 2)   # Back right
    ]
    #use rotation matrix to rotate points by car angle
    rotated_pts = []
    for x, y in pts:
        rx = x * np.cos(angle) - y * np.sin(angle)
        ry = x * np.sin(angle) + y * np.cos(angle)
        rotated_pts.append((pos[0] + rx, pos[1] + ry))

    pg.draw.polygon(screen, colour_palette['ORANGE'].value, rotated_pts)

def transform_pts(points, scale, offset):
    return [(int(p[0] * scale + offset[0]), int(p[1] * scale + offset[1])) for p in points]

def caculateAccelerations(vel, t):     
    a_arr = np.zeros(len(vel))
    for i in range(1, len(vel)):
        dt = t[i] - t[i-1]
        if dt == 0: #stop division by zero error
            a_arr[i] = 0
        else:            
            a_arr[i] = (vel[i] - vel[i-1]) / dt #change in vel / change in time
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
    fps = 60
    dt = 1/fps

    #screen = pg.display.set_mode(flags=pg.FULLSCREEN)
    screen = pg.display.set_mode((1280, 720))
    screen_shape = screen.get_size()
    pg.display.set_caption('Racing Lines')

    # Track data
    racing_line, best_time, vels, mesh = None, None, None, None
    pb_str = "00:00.000"
    best_time = [1]
    paused = False
    current_time = 0
    sim_time = 0


    # Background surface (static)
    background = pg.Surface(screen_shape)
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
    variables = default_variables.copy()
    track_dropdown = top_panel.track_name_display(variable_options['track'])

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
    follow_car_bool = False
    in_center_panel = False
    # Dividers
    dividers = BasePanel(screen_shape, font)



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
                    scale, offset = calculate_auto_scale(clip_rect, track_rect)
                    left_panel.reset()
                    right_panel.speed_graph.reset_telementry()

                if left_panel.pause_toggle.rect.collidepoint(event.pos):
                    paused = not paused
                
                if left_panel.follow_car.rect.collidepoint(event.pos):
                    follow_car_bool = not follow_car_bool
                
                if left_panel.save_ghost_btn.rect.collidepoint(event.pos):
                    right_panel.speed_graph.save_telementry()
                if left_panel.save_ghost_btn.rect.collidepoint(event.pos):
                    right_panel.speed_graph.update_ghost_bool()
                
                if right_panel.recalculate_btn.rect.collidepoint(event.pos):
                        top_panel.show_popup = True
                        top_panel.draw_popup(screen)
                        pg.display.flip()      
                        racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration = run_GA(variables, clip_rect)
                        top_panel.show_popup = False


            if event.type == pg.MOUSEMOTION:
                if clip_rect.collidepoint(event.pos): #over center panel
                    in_center_panel = True
                else: in_center_panel = False

            if event.type == pg.MOUSEWHEEL and in_center_panel and mesh is not None: #zoom
                current_total_offset = (offset[0] + pan[0], offset[1] + pan[1]) #total offset including pan
                mouse_pos = pg.mouse.get_pos()
                
                if event.y > 0: # zoom in
                    new_scale, new_total_offset = zoom(scale, current_total_offset, True, mouse_pos)
                else: # zoom out
                    new_scale, new_total_offset = zoom(scale, current_total_offset, False, mouse_pos)
                
                scale = new_scale
                offset = new_total_offset

                pan = (0, 0) #reset pan becasue it is now included into the offset
                

            left_panel.handle_event(event)
            track_dropdown.handle_event(event)
            right_panel.handle_event(event)
            top_panel.handle_event(event)
            

        if variables['tyre'] != right_panel.tyre_toggle.get_state():
            variables['tyre'] = right_panel.tyre_toggle.get_state()

        # track dropdown
        if variables['track'] != track_dropdown.get_track() and track_dropdown.get_track() != "Select Track":
            new_selection = track_dropdown.get_track()
            if new_selection == 'import': #if import is selected
                path, props_path = open_file_browser()
                print("path s", path)
                if path is not None: #if a file was selected
                    track_filename = Path(path).stem #get filename
                    add_real_properties(track_filename, props_path) #add real propeties to dict
                    track_dropdown.update_options(track_filename) #update dropdown options to include the new track and set it to the new track

                    variables['track'] = track_filename #set track to the new track
                    variables['custom_path'] = path #store custom track path for use in GA
                    changed_track = True
                else: 
                    if variables['track'] is None: #no previous track is selected
                        track_dropdown.set_track("Select Track")
                    else:
                        track_dropdown.set_track(variables['track']) #reset dropdwown to previous track if import cancelled so doesnt call open file browser again
                    print("Import cancelled")
                    changed_track = False
            else:
                variables['track'] = new_selection
                track_dropdown.set_track(new_selection)
                variables['custom_path'] = None
                changed_track = True

            if changed_track:
                pan = (0,0)
                top_panel.show_popup = True
                top_panel.draw_popup(screen)
                pg.display.flip()   
                left_panel.reset()     
                racing_line, best_time, vels, mesh, scale, offset, track_rect, pb_str, start_time, acceleration = run_GA(variables, clip_rect)
                right_panel.speed_graph.reset_telementry()
            
                right_panel.speed_graph.precompute_telemetry(best_time, vels)
                right_panel.GForce_graph.precompute_telemetry(best_time, acceleration)
                top_panel.show_popup = False
            
            print("----")
            print(f"Track changed to: {variables['track']}")
            print("----")

        #time stuff
        if not paused:
            if right_panel.lap_no_slider.get_clicked():
                current_time = right_panel.lap_no_slider.get_value() * best_time[-1]
            else:
                current_time+= dt
        sim_time = current_time % best_time[-1]


        
        #DRAWING
        screen.blit(background, (0, 0))
        dividers.draw_dividers(screen)
        left_panel.draw(screen)
        right_panel.draw(screen, sim_time)
        

        # top panel overlays
        top_panel.time_display(screen, current_time)
        top_panel.lap_display(screen, current_time / best_time[-1])
        top_panel.pb_display(screen, pb_str)
        top_panel.weather_text(screen)
        top_panel.weather_display(screen)
        track_dropdown.draw(screen, font)

        screen.set_clip(clip_rect)
        if mesh is not None:
            #raw car coords
            car_world_pos, car_angle = get_car_position(sim_time, racing_line, best_time)
            
            if follow_car_bool:
                #car in center and track moves around it
                render_offset = [clip_rect.centerx - (car_world_pos[0] * scale), 
                                 clip_rect.centery - (car_world_pos[1] * scale)] #offset to render track with car in center of screen
                car_screen_pos = clip_rect.center
                
                mini_rect = pg.Rect(clip_rect.x + 10, clip_rect.y + 10, 200, 200) #mini map rect in top left of track area
                m_scale, m_offset = calculate_auto_scale(mini_rect, track_rect, padding=5)
                mini_data = {
                    'scale': m_scale,
                    'offset': m_offset,
                    'car_pos': car_world_pos,
                    'rect': mini_rect
                }
            else:
                # car moving around track
                render_offset = (offset[0] + pan[0], offset[1] + pan[1])
                car_screen_pos = (car_world_pos[0] * scale + render_offset[0], 
                                  car_world_pos[1] * scale + render_offset[1])
                mini_data = None #no minimap


            draw_track_elements(screen, mesh, racing_line, vels, scale, render_offset, mini_data)
            draw_car(screen, car_screen_pos, car_angle)
            throttle = get_currentThrottle(sim_time, best_time, acceleration)
            right_panel.update_health_bar_sliders(throttle, (1-throttle))
            left_panel.save_ghost_func(racing_line, vels, best_time)
            left_panel.show_ghost_func(screen, scale, render_offset)
        
        screen.set_clip(None)
        
        clock.tick(fps)
        pg.display.flip()


if __name__ == '__main__':
    main()