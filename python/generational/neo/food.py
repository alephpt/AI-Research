import random
import pygame

class Food:
    def __init__(self, w, h):
        self.x = random.randint(0, w - 10)
        self.y = random.randint(0, h - 10)
        self.energy = random.randint(1, 10)
        self.cost = self.energy * 2
        self.fitness = self.energy // 2
        self.satisfaction = self.energy // 4
        self.color = (0, 255, 0)
        self.size = 5

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.size, self.size))