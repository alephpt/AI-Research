from enum import Enum
from .settings import Settings
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
    Home = (128, 128, 128) ##=>> We are not implementing this until units can not die a little bit later
    # TODO: Add 'Home'                

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
    def __init__(self, idx, x, y, unit_type=UnitType.Available):
        self.type = unit_type
        self.x = x
        self.y = y
        self.size = Settings.CELL_SIZE.value
        self.idx = idx

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        return isinstance(other, Unit) and self.x == other.x and self.y == other.y and self.idx == other.idx
    
    def update(self):
        pass

    def index(self):
        return self.y * self.size + self.x

    def draw(self, screen):
        if self.type == UnitType.Available:
            pygame.draw.rect(screen, self.type.value, (self.x * self.size, self.y * self.size, self.size, self.size), 1)
            return
        
        if self.type in [UnitType.Male, UnitType.Female]:
            color = self.status.combine(self.type) if str(type(self.status)) == "Status" else self.type.value
            pygame.draw.rect(screen, color, (self.x * self.size, self.y * self.size, self.size, self.size))

            label = pygame.font.Font(None, 24).render(f"{self.type.name}", True, (255, 255, 255))
            lx_position = realPosition(self.x, self.size, label.get_width())
            ly_position = realPosition(self.y, self.size, label.get_height() * 2)

            state = pygame.font.Font(None, 16).render(f"{self.status}", True, (255, 255, 255))
            sx_position = realPosition(self.x, self.size, state.get_width())
            sy_position = realPosition(self.y, self.size, state.get_height() / 4)

            screen.blit(label, (lx_position, ly_position))
            screen.blit(state, (sx_position, sy_position))



