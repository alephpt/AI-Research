from enum import Enum
from DGL.cosmos import Log, LogLevel, Settings 
from DGL.cosmos.closet.color import Color, clamp
from DGL.society.unittype import UnitType
import pygame

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
        self.radius = Settings.UNIT_RADIUS.value
        self.size = Settings.CELL_SIZE.value
        self.idx = idx
        self.type = cell_type
        self.x, self.y = self.getXY()
        Log(LogLevel.INFO, "Cell", f" ~~ Creating Cell {idx} - received type {cell_type} ~~")
        Log(LogLevel.INFO, "Cell", f" ~~ \tx: {self.x}, y: {self.y} ~~")

    def xy(self):
        return self.x, self.y

    def getXY(self):
        if self.type == UnitType.CELL:
            #Log(LogLevel.ALERT, "Cell", f"Cell {self.idx} is a Cell")
            return self.idx % Settings.GRID_SIZE.value, self.idx // Settings.GRID_SIZE.value

        if self.type in [UnitType.Male, UnitType.Female]:
            Log(LogLevel.ALERT, "Cell", f"Cell {self.idx} is a Unit")
            return Settings.randomLocation()
        
        Log(LogLevel.ALERT, "Cell", f"Cell {self.idx} is a {self.type.name}")
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
            self.type.drawUnit(screen, self.x, self.y, self.size, self.radius, rect)
            self.type.labelUnit(self.idx, screen, self.x, self.y, self.size, self.state)