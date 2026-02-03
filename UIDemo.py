import pygame as pg
from pygame.locals import *
from ComponentModule.components import *
from colours import colour_palette
import time
#from set_up_track import OrderOfOperations
import numpy as np

from TrackProcessing2.genetic_algorithm import main as ga


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
        self._draw_text(surface, f"Lap: {lap_no:.2f}", self.x_margin + (self.even_spacing * 4), self.y)

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
    def __init__(self, screen_shape, font, font_size, throttle_val=0.45, brake_val=0.9):
        super().__init__(screen_shape, font, font_size)
        self.border_x = int(0.8 * screen_shape[0])
        self.x_margin = int(self.border_x + (screen_shape[0] // 128))
        panel_width = int(screen_shape[0] - self.border_x)
        self.panel_height = screen_shape[1]
        
        self.slider_width, self.slider_height = int(panel_width * 15/16), int(self.panel_height * 0.03)

        # throttle slider
        self.throttle_slider = Slider(self.x_margin, int(self.panel_height * 0.2), self.slider_width, self.slider_height, 0, 1, throttle_val, "Throttle", False)
        self.throttle_pos = (self.x_margin, self.panel_height // 6)

        # brake slider
        self.brake_slider = Slider(self.x_margin, int(self.panel_height * 0.3), self.slider_width, self.slider_height, 0, 1, brake_val, "Brake", False)
        self.brake_pos = (self.x_margin, int(self.panel_height * 4/15))
    
    def handle_event(self, throttle_val, brake_val):
        self.throttle_slider = Slider(self.x_margin, int(self.panel_height * 0.2), self.slider_width, self.slider_height, 0, 1, throttle_val, "Throttle", False)
        self.brake_slider = Slider(self.x_margin, int(self.panel_height * 0.3), self.slider_width, self.slider_height, 0, 1, brake_val, "Brake", False)



    def draw(self, surface):
        self._draw_text(surface, "Throttle", *self.throttle_pos)
        self._draw_text(surface, "Brake", *self.brake_pos)
        self.throttle_slider.draw(surface, self.font)
        self.brake_slider.draw(surface, self.font)
        return


#
def draw_track_elements(screen, mesh, racing_line, velocities,  scale, offset):

    left_boundary = mesh[:, 0, :]
    right_boundary = mesh[:, -1, :]

    scale, offset = calculate_auto_scale(mesh, screen.get_clip())
    
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

def get_car_position(current_time, racing_line, time_array, scale, offset):

    sim_time = current_time % time_array[-1]
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

def get_currentThrottle(current_time, time_array, acceleration_array):
    sim_time = current_time % time_array[-1]
    print(time_array.shape, acceleration_array.shape)
    current_acceleration = np.interp(sim_time, time_array, acceleration_array)
    if current_acceleration > 0:
        current_throttle = 1

    else: current_throttle = 0
    return current_throttle
# ---------- MAIN GAME ----------
def main():
    # Initialise screen
    pg.init()
    start_time = time.time()

    screen = pg.display.set_mode(flags=pg.FULLSCREEN)
    screen = pg.display.set_mode((1920,1080))
    screen_shape = screen.get_size()
    pg.display.set_caption('Racing Lines')

    #Track data
    racing_line, best_time, vels, mesh = None, None, None, None
    pb_str = "00:00.000"
    best_time = [1]


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
    #track_list = ["Monaco", "Silverstone", "Spa", "Interlagos", "Suzuka", "Yas Marina", "Import"]
    track_list = ["Monza", "Silverstone", "Qatar", "Import"]
    track_name = "Silverstone"
    track_dropdown = top_panel.track_name_display(track_name, track_list)

    #order_of_operations = OrderOfOperations(track_name, dark_mode=True)
    #order_of_operations.run()
    #track_image = order_of_operations.get_track_image()
    # Clipping rectangle for track image
    x = int(0.1 * screen_shape[0])
    y = screen_shape[1] // 6
    w = int(0.8 * screen_shape[0]) - x
    h = screen_shape[1] - y
    DIVIDER_THICKNESS = 3
    clip_rect = pg.Rect(x + DIVIDER_THICKNESS, y + DIVIDER_THICKNESS, w - DIVIDER_THICKNESS, h - DIVIDER_THICKNESS)

    # Dividers
    dividers = Dividers(screen_shape)



    # Event loop
    while True:
        for event in pg.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN and event.key == K_ESCAPE:
                return
            if event.type == MOUSEBUTTONDOWN:
                print(pg.mouse.get_pos())

            left_panel.handle_event(event)
            track_dropdown.handle_event(event)

        # track dropdown
        if track_name != track_dropdown.get_track():
            track_name = track_dropdown.get_track()
            print(f"Running GA for {track_name}...")
            racing_line, best_time, vels, mesh = ga(track_name.lower())
            
            scale, offset = calculate_auto_scale(mesh, clip_rect)
            pb_str = f"{int(best_time[-1]//60)}: {best_time[-1]%60:.3f}"
            start_time = time.time()
            acceleration = caculateAccelerations(vels, best_time)
            print("----")
            print(f"Track changed to: {track_name}")
            print("----")


        current_time = time.time() - start_time
        
        #DRAWING
        screen.blit(background, (0, 0))
        dividers.draw(screen)
        left_panel.draw(screen)
        right_panel.draw(screen)
        

        # top panel overlays
        top_panel.time_display(screen, current_time)
        top_panel.lap_display(screen, current_time / best_time[-1])
        top_panel.pb_display(screen, pb_str)
        top_panel.weather_display(screen, "Sunny")
        track_dropdown.draw(screen, font)

        screen.set_clip(clip_rect)
        if mesh is not None:
            draw_track_elements(screen, mesh, racing_line, vels, scale, offset)
            pos, angle = get_car_position(current_time, racing_line, best_time, scale, offset)
            draw_car(screen, pos, angle)
            throttle = get_currentThrottle(current_time, best_time, acceleration)
            right_panel.handle_event(throttle, (1-throttle))
            #pg.draw.circle(screen, colour_palette['ORANGE'].value, (pos), 9)
        screen.set_clip(None)
  


        pg.display.flip()


if __name__ == '__main__':
    main()
