from enum import Enum
from .utilities import Log
from .settings import LogLevel, Settings
import pygame

class CellType(Enum):
    '''
    Defines whether a Cell is a Male or Female Unit, or a 
    different type of Placement or Interaction, as defined by the the Azimuth.'''
    Reserved = (85, 85, 85) # This is to 'delimit' the map from spawning near the edge
    Available = (64, 64, 64)
    Male = (128, 0, 0)
    Female = (0, 0, 128)
    HUMAN = [Male, Female]
    Market = (128, 128, 0)
    Home = (128, 128, 128) 

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
    def __init__(self, idx, cell_type=CellType.Available):
        Log(LogLevel.INFO, "Cell", f"Creating New Cell {idx} with type {cell_type.name}")
        self.x = idx % Settings.GRID_SIZE.value
        self.y = idx // Settings.GRID_SIZE.value
        self.idx = idx
        self.type = cell_type if self.inBounds() else CellType.Reserved
        self.size = Settings.CELL_SIZE.value

    def inBounds(self):
        start = Settings.GRID_START.value
        end = Settings.GRID_END.value
        Log(LogLevel.WARNING, "Cell", f"Checking in bounds of {start, end} :: {self.x, self.y}")
        in_bounds = (start, start) < (self.x, self.y) < (end, end)

        Log(LogLevel.WARNING, "Cell", f"{self.idx} :: {self.x, self.y} " + "is in bounds" if {in_bounds} else "is out of bounds")
        return 

    @staticmethod # Constructor that returns a set of the cells in their default state, as a set
    def set():
        return {Cell(i) for i in range(Settings.TOTAL_GRID_COUNT.value)}
    
    def list():
        return [Cell(i) for i in range(Settings.TOTAL_GRID_COUNT.value)]

    def xy(self):
        return self.x, self.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __eq__(self, other):
        if not isinstance(other, Cell):
            Log(LogLevel.FATAL, "Cell", f"!! FATAL !!  Cannot compare Cell to {type(other)}.")

        # THIS IS NOT BROKEN IF IT FAILS.
        Log(LogLevel.INFO, "Cell", f"Comparing {self.type}-{self.idx} to {other.type}-{other.idx}. Cannot compare Cell to {type(other)}.")
        return isinstance(other, Cell) and self.x == other.x and self.y == other.y and self.idx == other.idx
    
    def update(self):
        pass

    def index(self):
        return self.y * self.size + self.x

    def draw(self, screen):
        rect = (self.x * self.size, self.y * self.size, self.size, self.size)

        if self.type in [CellType.Reserved, CellType.Available]:
            pygame.draw.rect(screen, self.type.value, rect, 1)
            return
        
        if self.type in [CellType.Male, CellType.Female, CellType.Market, CellType.Home]:
            pygame.draw.rect(screen, self.type.value, rect)

            label = pygame.font.Font(None, 24).render(f"{self.type.name}", True, (255, 255, 255))
            lx_position = realPosition(self.x, self.size, label.get_width())
            ly_position = realPosition(self.y, self.size, label.get_height() * 2)

            state = pygame.font.Font(None, 16).render(f"{self.state}", True, (255, 255, 255))
            sx_position = realPosition(self.x, self.size, state.get_width())
            sy_position = realPosition(self.y, self.size, state.get_height() / 4)

            screen.blit(label, (lx_position, ly_position))
            screen.blit(state, (sx_position, sy_position))
