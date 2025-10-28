import random 
import pygame as pg

class randomPath:
    def getRandomPath(inner_sample, mesh, lateral_divs):
        path = []

        for i in range(len(inner_sample) -1):
            randomNum = random.randint(0, lateral_divs-1)
            path.append(mesh[i][randomNum])
        return path
    
    def drawPath(surface, path):
        for i in range(len(path)-1):
            p1 = (path[i][0], path[i][1])
            p2 = (path[i+1][0], path[i+1][1])
            pg.draw.line(surface, (255, 255, 0), p1, p2, 3)
