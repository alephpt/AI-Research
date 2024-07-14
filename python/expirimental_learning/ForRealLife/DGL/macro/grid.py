from copy import deepcopy
from multiprocessing import Array, Pool
import random

from ..meso.market import Market
from ..meso.unit import Unit
from ..meso.agency import Home
from ..meso.agency import State
from DGL.micro import Cell, Settings, CellType, Log, LogLevel

idx_size = Settings.SPAWN_AREA.value
idx_start = Settings.GRID_START.value
grid_size = Settings.GRID_SIZE.value

def helper(t, n):
    '''
    Generates a set, and then attempts to dedplucate the set, by checking the index of each cell.'''
    generated = set([t(random.randint(idx_start, idx_size - 1)) for _ in range(n)])
    resolved = set()

    for cellA in generated:
        for cellB in generated:
            if cellA.idx == cellB.idx:
                while cellA.idx == cellB.idx:
                    cellA = t(random.randint(idx_start, idx_size - 1))
            
        resolved.add(cellA)
        
    return resolved

# The actions of the units depends more on the world around them, than what they are doing
class Grid:
    '''
    The Grid acts as the connection between the Macro level to the Meso and Micro levels.

    We could rename this 'Mesh' if we really wanted.. or "Roots" or "Network" or "Web" or "Nexus" or "Hub" or "Core" or "Matrix" or "System" or "Structure" or "Framework".. or "Scaffold" or "Lattice"
    '''
    def __init__(self):
        self.cells = []
        self.units = set()
        self.selected = None
        Log(LogLevel.INFO, "Grid", f"Spawning Grid with Bounds {idx_start} to {idx_size}");
        self.repopulate()

    def alive(self):
        pass

    def repopulate(self):
        self.cells = Cell.list()
        self.units, markets, homes = self.populate() # This could be wrapped into a static Cell function in combination with the Unit Type

        for unit in self.units:
            unit.markets = markets
            unit.home = homes 

    def drawCells(self):
        for cell in self.cells:
            cell.draw(self.screen)

    def updateGrid(self, selected):
        Log(LogLevel.DEBUG, "Grid", f"Updating Grid of size {len(self.cells)}")
        Log(LogLevel.DEBUG, "Grid", f"Population: {len(self.units)}")

        for unit in self.units:
            # Remove iterables from units list, to prevent excessive steps
            if unit.state == State.Dead:
                self.units.remove(unit)
                continue

            # Prevents us from selecting None.
            if selected is not None:
                unit.target = selected
                unit.target_direction = unit.target.xy()
                # We actually need to determine the target direction aka Action Step unit.target.xy()

            # Determine if the Unit is at a Market or Home
            if unit.type in [CellType.Male, CellType.Female]:
                # TODO: Determine if we are 'broke' or 'hungry'
                if self.cells[unit.index()].type == CellType.Market:
                    # We need to determine Hungry, Broke, Working, or Buying Food
                    Log(LogLevel.VERBOSE, "Grid", f"Unit {unit} is at Market {self.cells[unit.index()].idx}")
                elif self.cells[unit.index()].type == CellType.Home:
                    Log(LogLevel.VERBOSE, "Grid", f"Unit {unit} is at Home {self.cells[unit.index()].idx}")
                    
            unit.updateValues()

    def availableCell(self, idx):
        Log(LogLevel.VERBOSE, "Grid", f"Checking if {idx} is available.")
        return self.cells[idx].type in [CellType.Available, CellType.Reserved]

    def emplaceCells(self, cell_set, f):
        cell_copy = deepcopy(cell_set)

        for cell in cell_copy:
            while not self.availableCell(cell.idx):
                Log(LogLevel.DEBUG, "Grid", f"Duplicate {cell.type} found at {cell.idx}.")
                cell = f(random.randint(idx_start, idx_size))

            self.cells[cell.idx] = cell
        
        return cell_copy

    def populate(self):
        '''
        Iterates through the number of units, markets, and homes, and generates a set of each type.'''
        star_map = [(Unit, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            res = pool.starmap(helper, star_map)

        units, markets, homes = res

        units = self.emplaceCells(units, Unit)
        markets = self.emplaceCells(markets, Market)
        homes = self.emplaceCells(homes, Home)

        return units, markets, homes
        
