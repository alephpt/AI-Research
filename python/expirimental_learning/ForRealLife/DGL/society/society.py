import random
from multiprocessing import Pool
from DGL.cosmos import Log, LogLevel
from .unit import Unit
from .market import Market
from .agency import Home, State
from DGL.engine.cell import Cell
from .unittype import UnitType
from DGL.cosmos import Settings

idx_start = Settings.GRID_START.value
idx_end = Settings.GRID_END.value
grid_size = Settings.GRID_SIZE.value

def helper(t, n):
    '''
    Generates a set, and then attempts to dedplucate the set, by checking the index of each cell.'''
    generated = set([t(random.randint(idx_start, idx_start + idx_end - 1)) for _ in range(n)])
    resolved = set()

    for cellA in generated:
        for cellB in generated:
            if cellA.idx == cellB.idx:
                while cellA.idx == cellB.idx:
                    cellA = t(random.randint(idx_start, idx_end - 1))
            
        resolved.add(cellA)
        
    return resolved


class Society:
    def __init__(self):
        self.repopulate()

    def repopulate(self):
        self.units, self.markets, self.homes = self.populate()
        self.cells = Cell.list()
         # This could be wrapped into a static Cell function in combination with the Unit Type

        for unit in self.units:
            unit.markets = self.markets
            unit.home = self.homes 

    def populate(self):
        Log(LogLevel.INFO, "Grid", f"Grid Size: {grid_size},{grid_size}")
        Log(LogLevel.INFO, "Grid", f"Total Grid Count: {Settings.TOTAL_GRID_COUNT.value}")
        Log(LogLevel.INFO, "Grid", f"Spawn Area: ({idx_start}, {idx_start}) to ({idx_end}, {idx_end})")

        '''
        Iterates through the number of units, markets, and homes, and generates a set of each type.'''
        star_map = [(Unit, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            res = pool.starmap(helper, star_map)

        units, markets, homes = res

        return units, markets, homes

    def updatePopulation(self, selected):
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
            if unit.type in [UnitType.Male, UnitType.Female]:
                # TODO: Determine if we are 'broke' or 'hungry'
                if self.cells[unit.index()].type == UnitType.Market:
                    # We need to determine Hungry, Broke, Working, or Buying Food
                    Log(LogLevel.VERBOSE, "Grid", f"Unit {unit} is at Market {self.cells[unit.index()].idx}")
                elif self.cells[unit.index()].type == UnitType.Home:
                    Log(LogLevel.VERBOSE, "Grid", f"Unit {unit} is at Home {self.cells[unit.index()].idx}")
                    
            unit.updateValues()
