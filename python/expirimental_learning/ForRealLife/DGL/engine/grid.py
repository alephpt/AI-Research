
from .cell import Cell, CellType
from DGL.cosmos import Log, LogLevel

# The actions of the units depends more on the world around them, than what they are doing
class Grid:
    '''
    The Grid acts as the connection between the Macro level to the Meso and Micro levels.

    We could rename this 'Mesh' if we really wanted.. or "Roots" or "Network" or "Web" or "Nexus" or "Hub" or "Core" or "Matrix" or "System" or "Structure" or "Framework".. or "Scaffold" or "Lattice"
    '''
    def __init__(self):
        '''
        We only care about the Cells and the Units, because anything else is referenced
        by the units, or cells, without regard for the other elements,
        '''
        self.cells = Cell.list()
        self.selected = None

    def alive(self):
        pass

    def drawCells(self):
        for cell in self.cells:
            cell.draw(self.screen)

    def availableCell(self, idx):
        Log(LogLevel.VERBOSE, "Grid", f"Checking if {idx} is available.")
        return self.cells[idx].type == CellType.Available