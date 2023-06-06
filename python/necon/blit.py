
import math
import pygame

class Circle:
    def __init__(self, x, y, radius, color):
        self.x, self.y = x, y
        self.radius = radius
        self.color = color
        
        
class Triangle:
    def __init__(self, location, size, color):
        self.x, self.y = location
        self.rotation = 0
        self.width, self.height = size
        self.p1, self.p2, self.p3 = (self.x, self.y - self.height/2), (self.x - self.width/2, self.y + self.height/2), (self.x + self.width/2, self.y + self.height/2)
        self.color = color
  
    def update(self):
        self.p1 = (self.x + math.cos(self.rotation) * self.height/2, self.y + math.sin(self.rotation) * self.height/2)
        self.p2 = (self.x + math.cos(self.rotation + 2 * math.pi/3) * self.height/2, self.y + math.sin(self.rotation + 2 * math.pi/3) * self.height/2)
        self.p3 = (self.x + math.cos(self.rotation + 4 * math.pi/3) * self.height/2, self.y + math.sin(self.rotation + 4 * math.pi/3) * self.height/2)
        
  
    def rotate(self, angle):
        self.rotation += angle
        self.rotation %= 2 * math.pi
        self.update()
    
    def move(self, x, y):
        print("moving player: " + str(x) + ", " + str(y))
        self.y += y * math.sin(self.rotation)
        self.rotate(x * math.pi/180)
        self.update()
        
    def draw(self, screen):
        print("drawing player: \nlocation: {}\npoints: {}\nrotation: {}".format(
            (self.x, self.y),
            (self.p1, self.p2, self.p3),
            self.rotation))
        pygame.draw.lines(screen, self.color, True, (self.p1, self.p2, self.p3), 1)