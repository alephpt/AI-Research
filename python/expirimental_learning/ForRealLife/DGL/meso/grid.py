from copy import deepcopy
from multiprocessing import Array, Pool
import random

from .market import Market
from .agent import Agent
from .agency import Home
from .agency import State
from DGL.micro import Unit, Settings, UnitType, Log, LogLevel

idx_size = Settings.GRID_SIZE.value ** 2
grid_size = Settings.GRID_SIZE.value


def helper(t, n):
    generated = set([t(random.randint(0, idx_size - 1)) for _ in range(n)])
    resolved = set()

    Log(LogLevel.VERBOSE, f"Generated {len(generated)} {t.__name__}s.")

    for unitA in generated:
        for unitB in generated:
            if unitA.idx == unitB.idx:
                Log(LogLevel.DEBUG, f"Duplicate {t.__name__}s found at {unitA.idx}.")
                while unitA.idx == unitB.idx:
                    unitA = t(random.randint(0, idx_size - 1))
            
        resolved.add(unitA)
        
    return resolved



# The actions of the agents depends more on the world around them, than what they are doing
class Grid:
    def __init__(self):
        self.cells = []
        self.agents = set()
        self.markets = set()
        self.homes = set()
        self.repopulate()

    def alive(self):
        return [agent for agent in self.agents if agent.state != State.Dead]

    def repopulate(self):
        self.cells = [Unit(i) for i in range(idx_size)]
        self.agents, self.markets, self.homes = self.populate()

    def draw(self, screen):
        for cell in self.cells:
            cell.draw(screen)

    def update(self):
        #Log(LogLevel.DEBUG, f"Updating Grid of size {len(self.cells)}")
       # Log(LogLevel.DEBUG, f"Population: {len(self.agents)}")

        for agent in self.agents:
            # Remove iterables from agents list, to prevent excessive steps
            if agent.state == State.Dead:
                self.agents.remove(agent)
                continue
                
            agent.updateValues()

    def availableCell(self, idx):
        return self.cells[idx].type == UnitType.Available

    def sanitizeSet(self, unit_set, unit_type, f):
        '''
        This helps to deduplicate indexes in a set of units, by checking for duplicates and replacing them.'''
        new_set = set()

        for unit in unit_set:
            if isinstance(unit_type, UnitType) and unit.type == unit_type or \
                isinstance(unit_type, list) and unit.type in unit_type:
                while not self.availableCell(unit.idx):
                    Log(LogLevel.DEBUG, f"Duplicate {unit_type} found at {unit.idx}.")
                    unit = f(random.randint(0, idx_size - 1))

            new_set.add(unit)

        return new_set

    def dedupliPlace(self, unit_set):
        '''
        This is a parallelized buffer for the sanitizing a generated set of units.

        This iterates through the generated set of Agends, Markets and Homes, and checks for duplicate indexes, in parallel.
        If a duplicate is found, it generates a new unit, and replaces the duplicate. with the help of the sanitizeSet method.
        '''
        agents, markets, homes = deepcopy(unit_set)
        star_map = [(agents, UnitType.HUMAN, Agent), (markets, UnitType.Market, Market), (homes, UnitType.Home, Home)]

        with Pool() as pool:
            res = pool.starmap(self.sanitizeSet, star_map)
            
        return res
    
    def emplaceUnits(self, unit_set, f):
        for unit in unit_set:
            while not self.availableCell(unit.idx):
                Log(LogLevel.DEBUG, f"Duplicate {unit.type} found at {unit.idx}.")
                unit = f(random.randint(0, idx_size - 1))

            self.cells[unit.idx] = unit

    def populate(self):
        '''
        Iterates through the number of agents, markets, and homes, and generates a set of each type.'''
        star_map = [(Agent, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            res = pool.starmap(helper, star_map)

        self.agents, self.markets, self.homes = self.dedupliPlace(res)

        self.emplaceUnits(self.agents, Agent)
        self.emplaceUnits(self.markets, Market)
        self.emplaceUnits(self.homes, Home)

        return res
        
