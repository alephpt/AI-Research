import pygame

from .status import Status
from .settings import Settings

def realPosition(azimuth, size, offset): 
    return (azimuth * size) + (size / 2) - (offset / 2)

class Placement:
    def __init__(self, x, y, unit_type):
        self.x = x
        self.y = y
        self.size = Settings.CELL_SIZE.value
        self.unit_type = unit_type
        self.status = 'static'

    def update(self):
        pass

    def index(self):
        return self.y * self.size + self.x

    def draw(self, screen):
        color = self.status.combine(self.unit_type) if type(self.status) == Status else self.unit_type.value
        pygame.draw.rect(screen, color, (self.x * self.size, self.y * self.size, self.size, self.size))

        label = pygame.font.Font(None, 24).render(f"{self.unit_type.name}", True, (255, 255, 255))
        lx_position = realPosition(self.x, self.size, label.get_width())
        ly_position = realPosition(self.y, self.size, label.get_height() * 2)

        state = pygame.font.Font(None, 16).render(f"{self.status}", True, (255, 255, 255))
        sx_position = realPosition(self.x, self.size, state.get_width())
        sy_position = realPosition(self.y, self.size, state.get_height() / 4)

        screen.blit(label, (lx_position, ly_position))
        screen.blit(state, (sx_position, sy_position))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    def __str__(self):
        return f"({self.x}, {self.y}) - {self.unit_type}"