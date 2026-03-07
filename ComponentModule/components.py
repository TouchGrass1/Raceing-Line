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
    def __init__(self, x, y, screen_shape, label, size='normal'):
        #button shape
        if size == 'normal':
            self.width = 150/1920 * screen_shape[0]
            self.height = 30/1080 * screen_shape[1]
        elif size == 'circle':
            self.height = 30/1080 * screen_shape[1]
            self.width = self.height
        
        self.rect = pg.Rect(x, y, self.width, self.height)

        self.clicked = False #check if the button was clicked, resets after get_clicked is called
        self.clicking = False # To handle holding down the button
        self.hover = False #to handle hover state
        self.label = label #text on button

    def draw(self, screen, font):
        if self.clicking: colour = colour_pallete['blue'] #mouse is down on the button
        elif self.hover: colour = colour_pallete['red2'] #mouse if over the button
        else: colour = colour_pallete['red'] #default
        pg.draw.rect(screen, colour, self.rect, border_radius= 20)
        label_text = font.render(self.label, True, colour_pallete['white'])
        label_rect = label_text.get_rect(center=self.rect.center)
        screen.blit(label_text, label_rect)
        
    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicking = True #start tracking mouse hold
        if event.type == pg.MOUSEBUTTONUP:
            if self.clicking and self.rect.collidepoint(event.pos):
                self.clicked = True #register only if mouse is released on the button
            self.clicking = False

        self.check_hover(event)
 

    def check_hover(self, event):
        if event.type == pg.MOUSEMOTION:
            if self.rect.collidepoint(event.pos):
                self.hover = True
            else:
                self.hover = False

    def get_clicked(self):
        if self.clicked:
            self.clicked = False
            return True
        return False
    
class Dropdown(Button):
    def __init__(self, x, y, screen_shape, label, options):
        super().__init__(x, y, screen_shape, label) #super button
        self.options = options #list of options
        self.expanded = False 
        self.selected_option = label #current selected option
        self.x= x
        self.y = y
        #print('1', self.selected_option)
        self.hover_options = -1 #to track which option is being hovered, -1 means none
        self.option_rects = [pg.Rect(x, y + (i+1)*self.height, self.width, self.height) for i in range(len(options))]

    def draw(self, screen, font):
        if self.expanded:
            #draw main button
            pg.draw.rect(screen, colour_pallete['blue'], self.rect, border_top_left_radius= 20, border_top_right_radius= 20)
            label_text = font.render(self.label, True, colour_pallete['white'])
            label_rect = label_text.get_rect(center=self.rect.center)
            screen.blit(label_text, label_rect)
            #draw options
            for i, option in enumerate(self.options):
                if i == self.hover_options: colour = colour_pallete['red2'] #hover
                else: colour = colour_pallete['line grey']
                if i == len(self.options) - 1:
                    pg.draw.rect(screen, colour, self.option_rects[i], border_bottom_left_radius= 20, border_bottom_right_radius= 20) #last option has rounded corners
                else:
                    pg.draw.rect(screen, colour, self.option_rects[i])
                option_text = font.render(option, True, colour_pallete['white'])
                option_rect = option_text.get_rect(center=self.option_rects[i].center)
                screen.blit(option_text, option_rect)
        else: super().draw(screen, font)
                
    def handle_event(self, event):
        
        if event.type == pg.MOUSEBUTTONDOWN and event.button == 1: #left click
            if self.rect.collidepoint(event.pos) and not self.expanded: #click once on main button:
                self.expanded = True

            elif self.expanded: #check which option is clicked
                for i, rect in enumerate(self.option_rects):
                    if rect.collidepoint(event.pos):
                        self.selected_option = self.options[i]
                        self.expanded = False
                        print('2', self.selected_option)
        if event.type == pg.MOUSEBUTTONUP and not self.rect.collidepoint(event.pos):
                self.expanded = False
                
        #check for hover over options
        if event.type == pg.MOUSEMOTION and self.expanded:
            self.hover_options = -1
            for i, rect in enumerate(self.option_rects):
                if rect.collidepoint(event.pos):
                    self.hover_options = i
                    break

        self.check_hover(event) #over main button


    def get_track(self):
        return self.selected_option

    def set_track(self, track_name):
        self.selected_option = track_name
        self.label = track_name

    def update_options(self, new_option):
        if new_option not in self.options:
                        self.options.pop() #remove import since it should always be last
                        self.options.append(new_option) #add new option
                        self.options.append('import') #add import back
                        self.option_rects = [pg.Rect(self.x, self.y + (i+1)*self.height, self.width, self.height) for i in range(len(self.options))] #update rects
                        self.selected_option = new_option #select new option
                        self.label = new_option

class Toggle(Button):
    def __init__(self, x, y, screen_shape, states):
        super().__init__(x, y, screen_shape, states) #super button
        self.states = states
        self.current_state = 0
        self.label = self.states[self.current_state] #current state
    
    def change_state(self, event):

        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.clicking = True
        if event.type == pg.MOUSEBUTTONUP:
            if self.clicking and self.rect.collidepoint(event.pos):
                self.current_state = (self.current_state + 1) % len(self.states) #cycle through states if clicked
            self.clicking = False

        self.check_hover(event)
        self.label = self.states[self.current_state]

    def get_state(self):
        return self.states[self.current_state]
    
    def set_state(self, state_num):
        self.states[state_num]
    

        
class Slider:
    def __init__(self, x, y, w, h, min_val, max_val, initial_val, label, active):
        self.active = active
        self.rect = pg.Rect(x, y, w, h)
        
        #values
        self.min = min_val
        self.max = max_val
        self.val = initial_val
        self.oldval = initial_val

        self.label = label
        self.dragging = False
        self.handle_x = x + (initial_val - min_val)/(max_val - min_val) * w #posistion of handle
        self.font = pg.font.Font(None, 30)
        self.clicked = False
        

    def draw(self, screen, font):
        #draw slider track
        pg.draw.rect(screen, colour_pallete['subtle grey'], self.rect, border_radius=23)
        
        if self.active:        
            #draw handle
            pg.draw.circle(screen, colour_pallete['white'], 
                            (int(self.handle_x), self.rect.centery), 10)
            
            #draw labels
            label_text = font.render(self.label, True, colour_pallete['white'])
            screen.blit(label_text, (self.rect.x - 20, self.rect.y - 30))
            
            min_text = font.render(f"{self.min:.1f}", True, colour_pallete['white'])
            screen.blit(min_text, (self.rect.left - 40, self.rect.centery)) #add small offset to avoid everlapping with handle
            
            max_text = font.render(f"{self.max:.1f}", True, colour_pallete['white'])
            screen.blit(max_text, (self.rect.right + 10, self.rect.centery ))
            
            val_text = font.render(f"{self.val:.2f}", True, colour_pallete['white'])
            screen.blit(val_text, (self.handle_x - 15, self.rect.bottom + 5))
        
        #health bar sliders
        else:
            x = (self.val - self.min)/(self.max - self.min) * self.rect.width #calculate width of filled part based on value
            rect = pg.Rect(self.rect.left, self.rect.top, x, self.rect.height )
            pg.draw.rect(screen, colour_pallete['red2'], rect, border_radius=23)
            
            #draw labels
            val_text = self.font.render(f"{self.val*100}%", 1, colour_pallete['white']) #display value as %
            pos = (self.rect.right - val_text.get_width(), self.rect.bottom + val_text.get_height())
            screen.blit(val_text, pos)



    def handle_event(self, event):
        x, y = pg.mouse.get_pos()

        if self.rect.collidepoint(x, y):
            if event.type == MOUSEBUTTONDOWN:
                self.dragging = True                

        if event.type == MOUSEBUTTONUP and self.dragging:
            self.dragging = False
            self.clicked = True

        if self.dragging:           
            self.handle_x = np.clip(x, self.rect.left, self.rect.right)
            self.val = self.min + (self.handle_x - self.rect.left)/self.rect.width * (self.max - self.min)

    def get_clicked(self):
        if self.clicked:
            self.clicked = False
            return True
        return False

    def get_value(self):
        return self.val

class Text:
    def __init__(self, x, y, text, font_size):
        self.pos = (x, y)
        self.text = text
        self.font = pg.font.Font(None, font_size)
    def draw(self, background):
        text = self.font.render(str(self.text), 1, colour_pallete['white'])
        background.blit(text, self.pos)

class EntryBox:

    def __init__(self, x, y, w, h, font, pwd='asdf', is_password=False, placeholder='Type asdf to start'):
        self.rect = pg.Rect(x, y, w, h)
        self.color_inactive = colour_pallete['line grey']
        self.color_active = colour_pallete['white']
        self.color = self.color_inactive
        self.pwd = pwd
        self.is_password = is_password
        self.placeholder = placeholder
        self.text = ''
        self.font = font
        self.txt_surface = self.font.render(self.text, True, self.color)
        self.active = False
        self.txt_surface = self.font.render(self.placeholder, True, self.color)
        self.is_correct = False
    def handle_event(self, event):
        if event.type == pg.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
                self.color = self.color_active
            else:
                self.active = False
                self.color = self.color_inactive
        if event.type == pg.KEYDOWN:
            if self.active:
                if event.key == pg.K_RETURN:
                    print(self.text)
                    if self.text != '':
                        if self.is_password:
                            if self.text == self.pwd:
                                print("Correct Password")
                                self.text = ''
                                self.is_correct = True
                                return
                        else:                        
                            self.is_correct = True
                            return
                    
                elif event.key == pg.K_BACKSPACE:
                    self.text = self.text[:-1]
                else:
                    if len(self.text) < 10:
                        self.text += event.unicode
                self.txt_surface = self.font.render(self.text, True, self.color)
        if not self.active:
            if self.text == '':
                self.txt_surface = self.font.render(self.placeholder, True, self.color)
            else:
                self.txt_surface = self.font.render(self.text, True, self.color)
        self.is_correct = False

    def get_text(self):
        return self.text

    def draw(self, screen):
        screen.blit(self.txt_surface, (self.rect.x+5, self.rect.y+5))
        pg.draw.rect(screen, self.color, self.rect, 2)

class TelemetryGraph:
    def __init__(self, x, y, w, h, title):
        self.rect = pg.Rect(x, y, w, h)
        self.title = title
        self.points = []
        self.STEP_SIZE_SECONDS=0.01 #10ms
        self.smooth_vels = None
        self.ghost_vels = None
        self.ghost_bool = False
    
    def precompute_telemetry(self, time_arr, vels):

        self.total_time = time_arr[-1]
        self.num_samples = int(self.total_time / self.STEP_SIZE_SECONDS) #calculate number of samples with step_s in seconds
        
        self.smooth_times = np.linspace(0, self.total_time, self.num_samples) #smooth out times

        time_arr = time_arr[0: len(vels)] 
        self.smooth_vels = np.interp(self.smooth_times, time_arr, vels) #lerp to smooth at velocties
    
    def save_telementry(self):
        self.ghost_vels = self.smooth_vels
    
    def reset_telementry(self):
        self.ghost_vels = None
    
    def update_ghost_bool(self):
        self.ghost_bool = not self.ghost_bool
        

    def draw(self, surface, font, sim_time, WINDOW_SIZE_SECONDS = 10):
        if WINDOW_SIZE_SECONDS <= 0:
            WINDOW_SIZE_SECONDS = 10
        window_indices = int(WINDOW_SIZE_SECONDS / self.STEP_SIZE_SECONDS)
        if self.smooth_vels is None:
            return
        current_idx = int((sim_time / self.STEP_SIZE_SECONDS))

        if self.ghost_bool and self.ghost_vels is not None:
            max_v = max(np.maximum(self.smooth_vels, self.ghost_vels))
            print(max_v)

            if current_idx >= window_indices:
                slice_vels = self.smooth_vels[current_idx - window_indices : current_idx]
                slice_ghost = self.ghost_vels[current_idx - window_indices : current_idx]
            else: #make list circular
                part2 = self.smooth_vels[0 : current_idx]
                part1 = self.smooth_vels[-(window_indices - len(part2)):]
                slice_vels = np.concatenate([part1, part2])

                part2 = self.ghost_vels[0 : current_idx]
                part1 = self.ghost_vels[-(window_indices - len(part2)):]
                slice_ghost = np.concatenate([part1, part2])
        else:
            max_v = max(self.smooth_vels)

            if current_idx >= window_indices:
                slice_vels = self.smooth_vels[current_idx - window_indices : current_idx]

            else: #make list circular
                part2 = self.smooth_vels[0 : current_idx]
                part1 = self.smooth_vels[-(window_indices - len(part2)):]
                slice_vels = np.concatenate([part1, part2])



        #Drawing part
        pts = []
        ghost_pts = []
        for i, val in enumerate(slice_vels):
            rel_x = i / window_indices
            px = self.rect.x + (rel_x * self.rect.width)
                    
            norm_y = np.clip(val / max_v, 0, 1)
            py = self.rect.bottom - (norm_y * self.rect.height) #inverted for pygame
            pts.append((px, py))

            if self.ghost_bool and self.ghost_vels is not None:
                norm_y_ghost = np.clip(slice_ghost[i] / max_v, 0, 1)
                py = self.rect.bottom - (norm_y_ghost * self.rect.height) #inverted for pygame
                ghost_pts.append((px, py))


        #bg
        pg.draw.rect(surface, (30, 30, 30), self.rect)
        pg.draw.rect(surface, colour_pallete['line grey'], self.rect, 1) #border

        #draw lines
        pg.draw.lines(surface, colour_pallete['blue'], False, pts, 2)
        if self.ghost_bool and self.ghost_vels is not None: pg.draw.lines(surface, colour_pallete['red2'], False, ghost_pts, 1)

        #draw title
        title_surf = font.render(self.title, True, colour_pallete['white'])
        surface.blit(title_surf, (self.rect.x, self.rect.y - 25))

        #draw axis

        #draw current value
        current_speed_text = font.render(f"{slice_vels[-1]:.1f} m/s", True, colour_pallete['white'])
        surface.blit(current_speed_text, (self.rect.right, py))


# def draw_mesh_pygame(screen, mesh, scale_factor=1.0, offset=(0, 0)):
#     # Define colors
#     RED = (246, 32, 57)
#     GREEN = (41, 148, 82)
#     LINE_GREY = (42, 45, 49)

#     # 1. Draw the internal "ribs" of the mesh (the black lines in your plt code)
#     for row in mesh:
#         # Convert coordinates to integers for Pygame
#         pts = [(int(p[0] * scale_factor + offset[0]), 
#                 int(p[1] * scale_factor + offset[1])) for p in row]
#         if len(pts) > 1:
#             pg.draw.lines(screen, LINE_GREY, False, pts, 1)

#     # 2. Draw Left Boundary (Red)
#     left_boundary = mesh[:, 0, :]
#     left_pts = [(int(p[0] * scale_factor + offset[0]), 
#                  int(p[1] * scale_factor + offset[1])) for p in left_boundary]
#     pg.draw.lines(screen, RED, False, left_pts, 3)

#     # 3. Draw Right Boundary (Green)
#     right_boundary = mesh[:, -1, :]
#     right_pts = [(int(p[0] * scale_factor + offset[0]), 
#                   int(p[1] * scale_factor + offset[1])) for p in right_boundary]
#     pg.draw.lines(screen, GREEN, False, right_pts, 3)