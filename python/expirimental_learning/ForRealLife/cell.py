import pygame
from unit import Unit

class Cell:
    def __init__(self, x, y, size):
        self.x = x
        self.y = y
        self.state = Unit.Available
        self.size = size
    
    def update(self, occupancy):
        self.state = occupancy

    def draw(self, screen):
        pygame.draw.rect(screen, self.state.value, (self.x * self.size, self.y * self.size, self.size, self.size), 1)