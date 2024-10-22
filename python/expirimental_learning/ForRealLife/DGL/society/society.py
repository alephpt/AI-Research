from multiprocessing import Pool
from DGL.cosmos import Log, LogLevel, MoveAction
from .unit import Unit
from .home import Home
from .market import Market
from .agency import State
from .unittype import UnitType
from DGL.cosmos import Settings

idx_start = Settings.GRID_START.value
idx_end = Settings.GRID_END.value
grid_size = Settings.GRID_SIZE.value

def helper(t, n):
    '''
    Generates a set, and then attempts to dedplucate the set, by checking the index of each cell.'''
    generated = set([t(i) for i in range(n)])
    resolved = set()

    for cellA in generated:


        for cellB in generated:
            if cellA.idx != cellB.idx:
                while cellA.xy() == cellB.xy():
                    cellA = t(cellA.idx)
            
        resolved.add(cellA)
        
    return resolved


class Society:
    def __init__(self):
        self.units, self.markets, self.homes = self.populate()
        self.weaveUnits()

    def weaveUnits(self):
        for unit in self.units:
            unit.markets = self.markets
            unit.home = self.homes

    def repopulate(self):
        self.target_selection = None # Triggers a reset on the World level
        del self.units, self.markets, self.homes
        self.units, self.markets, self.homes = self.populate()
        self.weaveUnits()

    # THIS SHOULD BE FORBIDDEN.
    def checkSelected(self, unit):
        '''
        We are checking if a given unit matches the selection of the grid coordinates provided by the user input'''
        if not isinstance(self.target_selection, tuple):
            return

        # If we have the same XY as a given unit, then we select that unit
        if unit.xy() == self.target_selection:
            Log(LogLevel.VERBOSE, "Society", f"Selected {unit} at {unit.xy()}")
            self.target_selection = unit

    def drawUnits(self):
        Log(LogLevel.VERBOSE, "Society", " ~ Drawing Units ~")
        Log(LogLevel.VERBOSE, "Society", f"Unit Count: {len(self.units)}")
        Log(LogLevel.VERBOSE, "Society", f"Market Count: {len(self.markets)}")
        Log(LogLevel.VERBOSE, "Society", f"Home Count: {len(self.homes)}")
        for unit in self.units:
            unit.draw(self.screen)
            self.checkSelected(unit)
        
        for market in self.markets:
            market.draw(self.screen)
            self.checkSelected(market)
        
        for home in self.homes:
            home.draw(self.screen)
            self.checkSelected(home)

    def populate(self):
        Log (LogLevel.INFO, "Society", " ~ Populating Society ~")
        Log(LogLevel.INFO, "Society", f"Grid Size: {grid_size},{grid_size}")
        Log(LogLevel.INFO, "Society", f"Total Grid Count: {Settings.TOTAL_GRID_COUNT.value}")
        Log(LogLevel.INFO, "Society", f"Spawn Area: ({idx_start}, {idx_start}) to ({idx_end}, {idx_end})")
        Log(LogLevel.INFO, "Society", f"Population: {Settings.N_POPULATION.value}")
        Log(LogLevel.INFO, "Society", f"Markets: {Settings.N_JOBS.value}")
        Log(LogLevel.INFO, "Society", f"Homes: {Settings.N_HOUSES.value}")

        '''
        Iterates through the number of units, markets, and homes, and generates a set of each type.'''
        star_map = [(Unit, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            res = pool.starmap(helper, star_map)

        units, markets, homes = res

        return units, markets, homes

    def updateUnits(self, target_selection):
        Log(LogLevel.VERBOSE, "Society", f" ~ Updating Population ~")
        Log(LogLevel.VERBOSE, "Society", f"Grid of size {len(self.cells)}")
        Log(LogLevel.VERBOSE, "Society", f"Population: {len(self.units)}")

        for unit in self.units:
            # Remove iterables from units list, to prevent excessive steps
            if unit.state == State.Dead:
                continue

                # This hook exists for us to be able to update state via the GUI
            # Prevents us from selecting None, but allows us to select a target during the simulation.
            if target_selection not in [unit.target_selection, None] and not type(target_selection) == tuple:
                unit.target_selection = target_selection # This piece of code updates all units to focus on a single selected targe
                    
            unit.updateAzimuth()
            unit.updateState()
            #unit.updateReward()
            unit.updateEthics()
