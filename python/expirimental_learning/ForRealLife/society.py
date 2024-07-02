from placement import Placement
from agent import Agent, Status, Action, Target
from unit import Unit
from multiprocessing import Pool
from enum import Enum
import random


   ##############
   ## GENESIS ##
   #############

# Macros for Society Genesis
def createAgents(n_population, grid_size, cell_size):
    return [Agent(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1), cell_size, grid_size, random.choice([Unit.Male, Unit.Female])) \
        for _ in range(n_population)]

def createJobs(n_jobs, grid_size, cell_size):
    return [Placement(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1), cell_size, Unit.Work) \
        for _ in range(n_jobs)]

def createFood(n_food, grid_size, cell_size):
    return [Placement(random.randint(0, grid_size - 1), random.randint(0, grid_size - 1), cell_size, Unit.Food) \
        for _ in range(n_food)]

class Genesis(Enum):
    Agents = createAgents
    Jobs = createJobs
    Food = createFood

def _helper(fn, n, grid_size, cell_size):
    return fn(n, grid_size, cell_size)


# Implement our Dictionary of Placement Types for Society Evolution.

  ###############
  ## EVOLUTION ##
  ###############

# Will eventually host the genetic learning algorithm and epoch loop
class Society:
    def __init__(self, grid_size, cell_size, screen, n_population, n_jobs, n_food):
        self.screen = screen
        self.grid_size = grid_size
        self.cell_size = cell_size
        # TODO: Fix duplicate placements bug

        self.jobs, self.food, self.population = self.populate(n_population, n_jobs, n_food)
    
    def findTarget(self, target):
        if target == Target.Food:
            return random.choice(self.food)
        elif target == Target.Work:
            return random.choice(self.jobs)
        # TODO: Implement these later
        # elif target == Target.Mate:
        #     return 
        # elif target == Target.Home:
        #     return random.choice([agent for agent in self.population if agent.status == Status.Alive])
        else:
            return None

    def update(self):
        print('Updating Society')

        population_alive = all([agent.status != Status.Dead for agent in self.population])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            #self.jobs, self.food, self.population = self.repopulate()
            return

        for agent in self.population:
            print(agent)
            agent.update(self.findTarget)
        # for job in self.jobs:
        #     job.update()
        # for food in self.food:
        #     food.update

    # Main Generator
    def populate(self, n_population, n_jobs, n_food):
        agency = (Genesis.Agents, n_population)
        employment = (Genesis.Jobs, n_jobs)
        sustenance = (Genesis.Food, n_food)

        with Pool() as pool:
            args = [(fn, arg, self.grid_size, self.cell_size) for fn, arg in (employment, sustenance, agency)]
            results = pool.starmap(_helper, args)
            print(results)

        jobs, food, population = results
        return jobs, food, population
        

    def draw(self):
        #print('Drawing Society')
        for job in self.jobs:
            job.draw(self.screen)
        for food in self.food:
            food.draw(self.screen)
        for agent in self.population:
            agent.draw(self.screen)