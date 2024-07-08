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
    '''
    Generates a set, and then attempts to dedplucate the set, by checking the index of each unit.'''
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

            if agent.type in [UnitType.Male, UnitType.Female]:
                # TODO: Determine if we are 'broke' or 'hungry'
                if self.cells[agent.index()].type == UnitType.Market:
                    # We need to determine Hungry, Broke, Working, or Buying Food
                    Log(LogLevel.INFO, f"Agent {agent} is at Market {self.cells[agent.index()].idx}")
                elif self.cells[agent.index()].type == UnitType.Home:
                    Log(LogLevel.INFO, f"Agent {agent} is at Home {self.cells[agent.index()].idx}")
                    
            agent.updateValues()

    def availableCell(self, idx):
        return self.cells[idx].type == UnitType.Available

    def emplaceUnits(self, unit_set, f):
        unit_copy = deepcopy(unit_set)

        for unit in unit_copy:
            while not self.availableCell(unit.idx):
                Log(LogLevel.DEBUG, f"Duplicate {unit.type} found at {unit.idx}.")
                unit = f(random.randint(0, idx_size - 1))

            self.cells[unit.idx] = unit
        
        return unit_copy

    def populate(self):
        '''
        Iterates through the number of agents, markets, and homes, and generates a set of each type.'''
        star_map = [(Agent, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            res = pool.starmap(helper, star_map)

        agents, markets, homes = res

        agents = self.emplaceUnits(agents, Agent)
        markets = self.emplaceUnits(markets, Market)
        homes = self.emplaceUnits(homes, Home)

        return agents, markets, homes
        
