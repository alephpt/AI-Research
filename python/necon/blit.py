
import math
import pygame

class Shape:
    def __init__(self, location, size, bounds, color):
        self.x, self.y = location
        self.max_x, self.max_y = bounds
        self.width, self.height = size
        self.rotation = 0
        self.rotation_speed = 0
        self.max_rotation_speed = 5
        self.velocity = 0
        self.max_velocity = 10
        self.p1 = (self.x, self.y - (self.height / 2))
        self.p2 = (self.x + (self.width / 2), self.y + (self.height / 2))
        self.p3 = (self.x - (self.width / 2), self.y + (self.height / 2))
        self.color = color
    
    def check_bounds(self):
        # check if the shape has hit the edge of the screen
        if self.p1[0] > self.max_x:
            self.x -= self.p1[0] - self.max_x
            self.velocity = 0
        if self.p2[0] > self.max_x:
            self.x -= self.p2[0] - self.max_x
            self.velocity = 0
        if self.p3[0] > self.max_x:
            self.x -= self.p3[0] - self.max_x
            self.velocity = 0
        if self.p1[0] < 0:
            self.x -= self.p1[0]
            self.velocity = 0
        if self.p2[0] < 0:
            self.x -= self.p2[0]
            self.velocity = 0
        if self.p3[0] < 0:
            self.x -= self.p3[0]
            self.velocity = 0
        if self.p1[1] > self.max_y:
            self.y -= self.p1[1] - self.max_y
            self.velocity = 0
        if self.p2[1] > self.max_y:
            self.y -= self.p2[1] - self.max_y
            self.velocity = 0
        if self.p3[1] > self.max_y:
            self.y -= self.p3[1] - self.max_y
            self.velocity = 0
        if self.p1[1] < 0:
            self.y -= self.p1[1]
            self.velocity = 0
        if self.p2[1] < 0:
            self.y -= self.p2[1]
            self.velocity = 0
        if self.p3[1] < 0:
            self.y -= self.p3[1]
            self.velocity = 0
    
    # updates p1, p2, p3 based on the current center x,y and rotation
    def update_points(self):
        self.p1 = (self.x, self.y - (self.height / 2))
        self.p2 = (self.x + (self.width / 2), self.y + (self.height / 2))
        self.p3 = (self.x - (self.width / 2), self.y + (self.height / 2))
        
        # rotate p1, p2, p3 around the center x,y
        self.p1 = (self.x + math.cos(self.rotation) * (self.p1[0] - self.x) - math.sin(self.rotation) * (self.p1[1] - self.y), self.y + math.sin(self.rotation) * (self.p1[0] - self.x) + math.cos(self.rotation) * (self.p1[1] - self.y))
        self.p2 = (self.x + math.cos(self.rotation) * (self.p2[0] - self.x) - math.sin(self.rotation) * (self.p2[1] - self.y), self.y + math.sin(self.rotation) * (self.p2[0] - self.x) + math.cos(self.rotation) * (self.p2[1] - self.y))
        self.p3 = (self.x + math.cos(self.rotation) * (self.p3[0] - self.x) - math.sin(self.rotation) * (self.p3[1] - self.y), self.y + math.sin(self.rotation) * (self.p3[0] - self.x) + math.cos(self.rotation) * (self.p3[1] - self.y))
    
        self.check_bounds()
    
    def accelerate(self, amount):
        self.velocity -= amount / 100
        if self.velocity > self.max_velocity:
            self.velocity = self.max_velocity
        if self.velocity < -self.max_velocity:
            self.velocity = -self.max_velocity
    
    def decelerate(self):
        if self.velocity > 0:
            self.velocity -= 0.01
        if self.velocity < 0:
            self.velocity += 0.01
    
    def turn(self, amount):
        self.rotation_speed += amount / 100
        if self.rotation_speed > self.max_rotation_speed:
            self.rotation_speed = self.max_rotation_speed
        if self.rotation_speed < -self.max_rotation_speed:
            self.rotation_speed = -self.max_rotation_speed
    
    def unturn(self):
        if self.rotation_speed > 0:
            self.rotation_speed -= 0.01
        if self.rotation_speed < 0:
            self.rotation_speed += 0.01
    
    def move(self, forward):
        self.y -= math.cos(self.rotation) * -forward
        self.x += math.sin(self.rotation) * -forward
        
        self.update_points()
    
    def rotate(self, angle):
        self.rotation += angle / 180 * math.pi
        self.update_points()
    
    def draw(self, screen):
        self.rotate(self.rotation_speed)
        self.move(self.velocity)
        pygame.draw.lines(screen, self.color, True, [self.p1, self.p2, self.p3], 1)