from copy import deepcopy
from multiprocessing import Pool
import random

from .market import Market
from .agent import Agent
from .agency import Home
from .agency import State
from DGL.micro import Unit, Settings, UnitType, Log, LogLevel

idx_size = Settings.GRID_SIZE.value ** 2
grid_size = Settings.GRID_SIZE.value


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

        for agent in self.agents:
            agent.draw(screen)

    def update(self):
        #Log(LogLevel.DEBUG, f"Updating Grid of size {len(self.cells)}")
       # Log(LogLevel.DEBUG, f"Population: {len(self.agents)}")

        for agent in self.agents:
            # Remove iterables from agents list, to prevent excessive steps
            if agent.state == State.Dead:
                self.agents.remove(agent)
                continue
                
            agent.updateValues()

    def helper(self, t, n):
        generated = [t(random.randint(0, idx_size - 1)) for _ in range(n)]
        
        Log(LogLevel.VERBOSE, f"Generated {len(generated)} {t.__name__}s")
        Log(LogLevel.VERBOSE, f"{len(self.cells)} cells in grid of size {grid_size}x{grid_size}")

        for unit in generated:
            Log(LogLevel.VERBOSE, f"Placing {unit.type} at {unit.x}, {unit.y} with index {unit.idx}")
            
            # We make sure we have an available cell to place the unit
            while self.cells[unit.idx].type != UnitType.Available:
                unit = t(random.randint(0, idx_size - 1))
                Log(LogLevel.VERBOSE, f"Retrying placement of {unit.type} at {unit.x}, {unit.y} with index {unit.idx}")
            
            # We place the unit in the cell
            self.cells[unit.idx] = unit

        return generated

    def populate(self):
        star_map = [(Agent, Settings.N_POPULATION.value), (Market, Settings.N_JOBS.value), (Home, Settings.N_HOUSES.value)]

        with Pool() as pool:
            res = pool.starmap(self.helper, star_map)
            #pool.close()

        return res
        
