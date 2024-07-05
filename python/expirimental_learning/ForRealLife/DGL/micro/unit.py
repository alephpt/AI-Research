from enum import Enum

from .settings import LogLevel, Settings
import pygame

class UnitType(Enum):
    '''
    Defines whether a Unit is a Male or Female Agent, or a 
    different type of Placement or Interaction, as defined by the the Azimuth.'''
    Available = (64, 64, 64)
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    Work = (128, 128, 0)
    Food = (0, 128, 0)
    Home = (128, 128, 128) 

def realPosition(azimuth, size, offset): 
    return (azimuth * size) + (size / 2) - (offset / 2)

class Unit:
    '''
    Defines whether a Unit is a Male or Female Agent, or a 
    different type of Placement or Interaction, as defined by the the Azimuth.
    
    Init Parameters:
    idx: int
    x: int
    y: int
    unit_type: Unit = Unit.Available
    '''
    def __init__(self, idx, unit_type=UnitType.Available):
        self.type = unit_type
        self.x = idx % Settings.GRID_SIZE.value
        self.y = idx // Settings.GRID_SIZE.value
        self.idx = idx
        self.size = Settings.CELL_SIZE.value

    @staticmethod # Constructor that returns a set of the units in their default state, as a set
    def set():
        return {Unit(i) for i in range(Settings.GRID_SIZE.value ** 2)}

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        print("[Unit] :: Comparing {self} to {other}")
        return isinstance(other, Unit) and self.x == other.x and self.y == other.y and self.idx == other.idx
    
    def update(self):
        pass

    def index(self):
        return self.y * self.size + self.x

    def draw(self, screen):
        rect = (self.x * self.size, self.y * self.size, self.size, self.size)

        if self.type == UnitType.Available:
            pygame.draw.rect(screen, self.type.value, rect, 1)
            return
        
        if self.type in [UnitType.Male, UnitType.Female, UnitType.Work, UnitType.Food]:
            pygame.draw.rect(screen, self.type.value, rect)

            label = pygame.font.Font(None, 24).render(f"{self.type.name}", True, (255, 255, 255))
            lx_position = realPosition(self.x, self.size, label.get_width())
            ly_position = realPosition(self.y, self.size, label.get_height() * 2)

            state = pygame.font.Font(None, 16).render(f"{self.state}", True, (255, 255, 255))
            sx_position = realPosition(self.x, self.size, state.get_width())
            sy_position = realPosition(self.y, self.size, state.get_height() / 4)

            screen.blit(label, (lx_position, ly_position))
            screen.blit(state, (sx_position, sy_position))

