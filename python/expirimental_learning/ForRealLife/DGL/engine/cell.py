from enum import Enum
from DGL.cosmos import Log, LogLevel, Settings
from DGL.society.unittype import UnitType
import pygame

def realPosition(azimuth, size, offset): 
    return (azimuth * size) + (size / 2) - (offset / 2)


class Cell:
    '''
    Defines whether a Cell is a Male or Female Unit, or a 
    different type of Placement or Interaction, as defined by the the Azimuth.
    
    Init Parameters:
    idx: int
    x: int
    y: int
    cell_type: Cell = Cell.Available
    '''
    def __init__(self, idx, cell_type=UnitType.CELL):
        #Log(LogLevel.ALERT, "Cell", f" ~~ Creating Cell {idx} - received type {cell_type} ~~")
        self.radius = Settings.UNIT_RADIUS.value
        self.size = Settings.CELL_SIZE.value
        self.idx = idx
        self.type = cell_type
        self.x, self.y = self.getXY()

        #Log(LogLevel.ALERT, "Cell", f" ~~ New Cell {self.idx} created with type {self.type.name} ~~ \n")

    def xy(self):
        return self.x, self.y

    def getXY(self):
        if self.type == UnitType.CELL:
            return self.idx % Settings.GRID_SIZE.value, self.idx // Settings.GRID_SIZE.value

        if self.type in [UnitType.Male, UnitType.Female]:
            return Settings.randomLocation()
        return Settings.randomWithinBounds()

    @staticmethod # Constructor that returns a set of the cells in their default state, as a set
    def set():
        return {Cell(i) for i in range(Settings.TOTAL_GRID_COUNT.value)}
    
    def list():
        return [Cell(i) for i in range(Settings.TOTAL_GRID_COUNT.value)]

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if other is None:
            return False

        if not isinstance(other, Cell):
            Log(LogLevel.FATAL, "Cell", f"!! FATAL !!  Cannot compare {self.type} to {type(other)}.")

        # THIS IS NOT BROKEN IF IT FAILS.
        Log(LogLevel.INFO, "Cell", f"Comparing {self.type}-{self.idx} to {other.type}-{other.idx}. Cannot compare Cell to {type(other)}.")
        return isinstance(other, Cell) and self.x == other.x and self.y == other.y and self.idx == other.idx
    
    def update(self):
        pass

    def index(self):
        return self.y * self.size + self.x

    def draw(self, screen):
        rect = (self.x * self.size, self.y * self.size, self.size, self.size)
        #Log(LogLevel.ALERT, "Cell", f"Drawing ~ {self.type.name}-{self.idx} at {self.x}, {self.y}")
        #Log(LogLevel.ALERT, "Cell", f"Drawing ~ \t\t [{rect}]")

        if self.type == UnitType.CELL:
            pygame.draw.rect(screen, self.type.color().value, rect, 1)
            return
        
        if self.type in [UnitType.Male, UnitType.Female, UnitType.Market, UnitType.Home]:
            pygame.draw.circle(screen, self.type.add((50, 50, 50)), (self.x * self.size + self.size / 2, self.y * self.size + self.size / 2), self.radius, 1)
            pygame.draw.rect(screen, self.type.color().value, rect)

            label = pygame.font.Font(None, 24).render(f"{self.type.name}-{self.idx}", True, (255, 255, 255))
            lx_position = realPosition(self.x, self.size, label.get_width())
            ly_position = realPosition(self.y, self.size, label.get_height() * 2)

            state = pygame.font.Font(None, 16).render(f"{self.state}", True, (255, 255, 255))
            sx_position = realPosition(self.x, self.size, state.get_width())
            sy_position = realPosition(self.y, self.size, state.get_height() / 4)

            screen.blit(label, (lx_position, ly_position))
            screen.blit(state, (sx_position, sy_position))
