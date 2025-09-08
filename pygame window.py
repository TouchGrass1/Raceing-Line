import pygame

import pygame_widgets
from pygame_widgets.button import Button

def find_track_boundaries():
    #Find Vertical and horizontal side
    prev_black = 0 #boolean if the previous colour was black or not
    for i in range 2:
    	order = [screen.width, screen.height]
    	for y in range order[0 + i]: #first does vertical side, then changes the order to do horizontal 
    		for x in range order[1 - i]:
    			if get_at((x, y)) == black:
    				if prev_black == 0:
    					set_at((x, y), green) #change track boundary to green
    					prev_black = 1
    			else: #pixel colour is white or orange
    				prev_black = 0
    				if get_at((x, y)) == white:
    					set_at((x, y), orange) #change all white to orange cuz mclaren and show that its been checked

# Set up Pygame
pygame.init()
screen = pygame.display.set_mode((600, 600))

# Creates the button with optional parameters
button = Button(
    # Mandatory Parameters
    screen,  # Surface to place button on
    100,  # X-coordinate of top left corner
    100,  # Y-coordinate of top left corner
    300,  # Width
    150,  # Height

    # Optional Parameters
    text='Hello',  # Text to display
    fontSize=50,  # Size of font
    margin=20,  # Minimum distance between text/image and edge of button
    inactiveColour=(200, 50, 0),  # Colour of button when not being interacted with
    hoverColour=(150, 0, 0),  # Colour of button when being hovered over
    pressedColour=(0, 200, 20),  # Colour of button when being clicked
    radius=20,  # Radius of border corners (leave empty for not curved)
    onClick=lambda: print('Click')  # Function to call when clicked on
)


run = True
while run:
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.QUIT:
            pygame.quit()
            run = False
            quit()

    screen.fill((255, 255, 255))

    pygame_widgets.update(events)  # Call once every loop to allow widgets to render and listen
    pygame.display.update()

