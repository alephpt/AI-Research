import pygame
from unit import Unit

class Grid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.units = []

    def add_unit(self, unit):
        self.units.append(unit)

    def update(self):
        for unit in self.units:
            if unit.alive:
                unit.update(self)
        self.units = [unit for unit in self.units if unit.alive]

    def draw(self, screen, generation, avg_age, max_age, min_age):
        screen.fill((255, 255, 255))
        for unit in self.units:
            color = (0, 0, 255) if unit.sex == 'M' else (255, 0, 0)
            pygame.draw.circle(screen, color, (unit.x * 12, unit.y * 8), 5)

        font = pygame.font.SysFont(None, 24)
        text = font.render(f'Generation: {generation}, Avg Age: {avg_age:.2f}, Max Age: {max_age}, Min Age: {min_age}', True, (0, 0, 0))
        screen.blit(text, (10, 10))
