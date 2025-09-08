import pygame


background_colour = (234, 212, 252)

screen = pygame.display.set_mode((300, 300))

pygame.display.set_caption('Geeksforgeeks')

screen.fill(background_colour)

# Update the display using flip
pygame.display.flip()


running = True


while running:
  
# for loop through the event queue  
    for event in pygame.event.get():
     
        if event.type == pygame.QUIT:

            running = False
