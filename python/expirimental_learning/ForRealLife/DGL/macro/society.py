import pygame
from .genesis import Genesis
from .dna import Genome
from DGL.meso import Status
from DGL.micro import Settings, Unit
import random

cell_size = Settings.CELL_SIZE.value
grid_size = Settings.GRID_SIZE.value

# Implement our Dictionary of Placement Types for Society Evolution.

  ###############
  ## EVOLUTION ##
  ###############

# Will eventually host the genetic learning algorithm and epoch loop
class Society(Genome):
    def __init__(self, screen):
        self.screen = screen
        super().__init__()
        self.flattened_cells = set()
        print('Creating Society with Grid Size:', grid_size)
        self.cells = [Unit(y * grid_size + x, x, y) for x in range(grid_size) for y in range(grid_size)]

        # TODO: Fix duplicate placements bug
        self.repopulate()

    def repopulate(self):
        self.population, self.jobs, self.food  = Genesis.creation(self.flattened_cells)
        

    def findTarget(self):
        return random.choice(self.food)

    def update(self):
        population_alive = all([agent.status != Status.Dead for agent in self.population])

        # We should only hit this if all agents are dead
        if not population_alive:
            print('Population is Dead')
            #self.jobs, self.food, self.population = self.repopulate()
            return

        # This is where we need to update the state of all of the cells
        for agent in self.population:
            agent.update(self.findTarget)
        
        for unit in self.cells:
                unit.update()
    
        self.updateStatistics()

    def drawStatus(self):
        sections = ["Status", "AvgAge", "AvgHealth", "AvgWealth", "AvgHappiness", "AvgReward"]
        values = [Status.fromValue(self.n_alive), self.avg_age, self.avg_health, self.avg_wealth, self.avg_happiness, self.avg_reward]
        font = pygame.font.Font(None, 22)

        # Transparent Frame
        new_surface = pygame.Surface((2.5 * cell_size, 7 * 22))
        new_surface.set_alpha(100)
        new_surface.fill((0, 0, 0))
        self.screen.blit(new_surface, ((grid_size - 2.5) * cell_size - 22, 11))

        height_offset = 24
        for i, (section, value) in enumerate(zip(sections, values)):
            section = font.render(f"{section}:", True, (222, 222, 222))
            value = font.render(f"{value}", True, (255, 255, 255))
            width_offset = value.get_width() + 16
            self.screen.blit(section, ((grid_size - 2.5) * cell_size, i * 22 + height_offset))
            self.screen.blit(value, (grid_size * cell_size - width_offset - 22, i * 22 + height_offset))



    def draw(self):
        #print('Drawing Society')
        for job in self.jobs:
            job.draw(self.screen)
        for food in self.food:
            food.draw(self.screen)
        for agent in self.population:
            agent.draw(self.screen)
        for unit in self.cells:
            unit.draw(self.screen)
