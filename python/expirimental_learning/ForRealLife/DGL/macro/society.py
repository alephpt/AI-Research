import pygame
from DGL.meso import Agent, Status, Target
from DGL.micro import Unit, Placement, Settings
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
    def __init__(self, screen, ):
        self.screen = screen

        # TODO: Fix duplicate placements bug
        self.jobs, self.food, self.population = self.populate()

    
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
        population_alive = all([agent.status != Status.Dead for agent in self.population])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            #self.jobs, self.food, self.population = self.repopulate()
            return

        for agent in self.population:
            #print(agent)
            agent.update(self.findTarget)
        # for job in self.jobs:
        #     job.update()
        # for food in self.food:
        #     food.update

        self.updateStatistics()

    # Main Generator
    def populate(self):
        agency = (Genesis.Agents, Settings.N_POPULATION.value)
        employment = (Genesis.Jobs, Settings.N_JOBS.value)
        sustenance = (Genesis.Food, Settings.N_FOOD.value)

        with Pool() as pool:
            args = [employment, sustenance, agency]
            results = pool.starmap(_helper, args)
            print(results)

        return results
    
    def drawStatus(self):
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward"]
        values = [Status.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward]
        font = pygame.font.Font(None, 22)

        # Transparent Frame
        new_surface = pygame.Surface((2.5 * self.cell_size, 7 * 22))
        new_surface.set_alpha(100)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, ((self.grid_size - 2.5) * self.cell_size - 22, 11))

        height_offset = 24
        for i, (section, value) in enumerate(zip(sections, values)):
            section = font.render(f"{section}:", True, (222, 222, 222))
            value = font.render(f"{value:.5}", True, (255, 255, 255))
            width_offset = value.get_width() + 16
            self.screen.blit(section, ((self.grid_size - 2.5) * self.cell_size, i * 22 + height_offset))
            self.screen.blit(value, (self.grid_size * self.cell_size - width_offset - 22, i * 22 + height_offset))



    def draw(self):
        #print('Drawing Society')
        for job in self.jobs:
            job.draw(self.screen)
        for food in self.food:
            food.draw(self.screen)
        for agent in self.population:
            agent.draw(self.screen)
