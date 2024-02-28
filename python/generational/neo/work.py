import random
import pygame

class Work:
    def __init__(self, w, h):
        self.x = random.randint(0, w - 10)
        self.y = random.randint(0, h - 10)
        self.energy = -random.randint(1, 20)
        self.pay = self.energy * -2
        self.satisfaction = self.energy ** 2
        self.fitness = self.satisfaction // 4
        self.color = (178, 178, 178)
        self.size = 10
        
    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size), 1)