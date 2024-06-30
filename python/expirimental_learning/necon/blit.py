
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
        self.velocity_x = 0
        self.velocity_y = 0
        self.max_velocity = 10
        self.p1 = (self.x, self.y - (self.height / 2))
        self.p2 = (self.x + (self.width / 2), self.y + (self.height / 2))
        self.p3 = (self.x - (self.width / 2), self.y + (self.height / 2))
        self.color = color
    
    def check_bounds(self):
        # check if the shape has hit the edge of the screen
        if self.p1[0] > self.max_x:
            self.x -= self.p1[0] - self.max_x
            self.velocity_x = 0
        if self.p2[0] > self.max_x:
            self.x -= self.p2[0] - self.max_x
            self.velocity_x = 0
        if self.p3[0] > self.max_x:
            self.x -= self.p3[0] - self.max_x
            self.velocity_x = 0
        if self.p1[0] < 0:
            self.x -= self.p1[0]
            self.velocity_x = 0
        if self.p2[0] < 0:
            self.x -= self.p2[0]
            self.velocity_x = 0
        if self.p3[0] < 0:
            self.x -= self.p3[0]
            self.velocity_x = 0
        if self.p1[1] > self.max_y:
            self.y -= self.p1[1] - self.max_y
            self.velocity_y = 0
        if self.p2[1] > self.max_y:
            self.y -= self.p2[1] - self.max_y
            self.velocity_y = 0
        if self.p3[1] > self.max_y:
            self.y -= self.p3[1] - self.max_y
            self.velocity_y = 0
        if self.p1[1] < 0:
            self.y -= self.p1[1]
            self.velocity_y = 0
        if self.p2[1] < 0:
            self.y -= self.p2[1]
            self.velocity_y = 0
        if self.p3[1] < 0:
            self.y -= self.p3[1]
            self.velocity_y = 0
    
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
        self.velocity_x += math.sin(self.rotation) * amount
        self.velocity_y += math.cos(self.rotation) * amount
        speed = math.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)
        if speed > self.max_velocity:
            scale = self.max_velocity / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
        if speed < -self.max_velocity / 2:
            scale = (-self.max_velocity / 2) / speed
            self.velocity_x *= scale
            self.velocity_y *= scale
    
    def decelerate(self):
        if self.velocity_x > 0:
            self.velocity_x -= 0.1
        if self.velocity_x < 0:
            self.velocity_x += 0.1
        if self.velocity_y > 0:
            self.velocity_y -= 0.1
        if self.velocity_y < 0:
            self.velocity_y += 0.1
    
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
    
    def move(self):
        self.y -= self.velocity_y
        self.x += self.velocity_x
        
        self.update_points()
    
    def rotate(self, angle):
        self.rotation += angle / 180 * math.pi
        self.update_points()
    
    def draw(self, screen):
        self.rotate(self.rotation_speed)
        self.move()
        pygame.draw.lines(screen, self.color, True, [self.p1, self.p2, self.p3], 1)