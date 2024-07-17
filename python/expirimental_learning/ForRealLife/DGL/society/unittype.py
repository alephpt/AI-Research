from enum import Enum
from DGL.cosmos.closet.color import Color
from DGL.cosmos import LogLevel, Log, realPosition
import pygame

class UnitType(Enum):
    CELL = Color.CELLULAR_GREEN
    Male = Color.UNIT_RED
    Female = Color.UNIT_BLUE
    Market = Color.Olive
    Home = Color.Brown

    def __str__(self):
        return self.name
    
    def color(self):
        return self.value
    
    def add(self, other):
        if isinstance(other, UnitType):
            return self.value.add(other.color())
        
        if isinstance(other, Color):
            return self.value.add(other)
        
        return self.color().add(other)
    
    def sub(self, other):
        if isinstance(other, UnitType):
            return self.value.sub(other.color())
        
        if isinstance(other, Color):
            return self.value.sub(other)
        
        return self.color().sub(other)
    
    def labelUnit(self, idx, screen, x, y, size, state):
        '''
        takes in an index, screen, x, y, size, and state
        '''
        label = pygame.font.Font(None, 24).render(f"{self.name}-{idx}", True, (255, 255, 255))
        lx_position = realPosition(x, size, label.get_width())
        ly_position = realPosition(y, size, label.get_height() * 2)

        state = pygame.font.Font(None, 16).render(f"{state}", True, (255, 255, 255))
        sx_position = realPosition(x, size, state.get_width())
        sy_position = realPosition(y, size, state.get_height() / 4)

        screen.blit(label, (lx_position, ly_position))
        screen.blit(state, (sx_position, sy_position))

    def drawUnit(self, screen, x, y, size, radius, rect):
        '''
        takes in a screen, x, y, size, and radius
        '''
        if self in [UnitType.Male, UnitType.Female]:
            radial_color = self.add((50, 50, 50)).sub(Color.CELLULAR_GREEN).value
            circle_location = (x * size + size // 2, y * size + size // 2) # TODO: Optimize out all Settings.HALF_SIZE values - Could precalculate all grid values
            
            pygame.draw.circle(screen, radial_color, circle_location, radius, 1)

        pygame.draw.rect(screen, self.color().value, rect)
